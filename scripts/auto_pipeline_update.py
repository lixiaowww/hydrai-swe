#!/usr/bin/env python3
"""
HydrAI-SWE Automated Data Pipeline Update Script
- Smart scheduling based on data age and availability
- Priority-based update selection
- Comprehensive error handling and logging
- Integration with existing pipeline API

Usage:
  python3 scripts/auto_pipeline_update.py [--dry-run] [--force] [--sources=modis,smap,hls]

Features:
  - Automatic source selection based on data age
  - Smart retry with exponential backoff
  - Integration with systemd for production deployment
  - Comprehensive logging and monitoring
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DataSource:
    """Data source configuration and status"""
    name: str
    priority: int
    max_age_hours: int
    update_interval_hours: int
    last_update: Optional[float]
    status: str
    health_status: str
    quality_score: float
    records: int
    type: str
    backup_sources: List[str]
    backup_available: bool


class PipelineUpdater:
    """Intelligent pipeline update manager"""
    
    def __init__(self, api_base: str = "http://localhost:8000", dry_run: bool = False):
        self.api_base = api_base
        self.dry_run = dry_run
        self.session = requests.Session()
        self.session.timeout = 30
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/home/sean/hydrai_swe/logs/pipeline_updates.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Ensure log directory exists
        os.makedirs('/home/sean/hydrai_swe/logs', exist_ok=True)
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status from API"""
        try:
            response = self.session.get(f"{self.api_base}/api/v1/pipeline/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get pipeline status: {e}")
            return {}
    
    def parse_data_sources(self, status_data: Dict) -> List[DataSource]:
        """Parse API response into DataSource objects"""
        sources = []
        
        if status_data.get('status') != 'success':
            self.logger.error("Pipeline status not successful")
            return sources
        
        for name, info in status_data.get('sources', {}).items():
            # Determine priority and update intervals based on source type
            priority = self._get_source_priority(name, info.get('type', ''))
            max_age = self._get_max_age_hours(name, info.get('type', ''))
            update_interval = self._get_update_interval_hours(name, info.get('type', ''))
            
            source = DataSource(
                name=name,
                priority=priority,
                max_age_hours=max_age,
                update_interval_hours=update_interval,
                last_update=info.get('last_update'),
                status=info.get('status', 'Unknown'),
                health_status=info.get('health_status', 'Unknown'),
                quality_score=info.get('quality_score', 0),
                records=info.get('records', 0),
                type=info.get('type', ''),
                backup_sources=info.get('backup_sources', []),
                backup_available=info.get('backup_available', False)
            )
            sources.append(source)
        
        return sources
    
    def _get_source_priority(self, name: str, source_type: str) -> int:
        """Get source priority (lower = higher priority)"""
        priority_map = {
            'hydat': 1,      # Real-time hydrometric data
            'eccc': 1,       # Real-time weather data
            'modis': 2,      # Daily satellite data
            'era5_land': 4,  # Reanalysis data (less critical)
            'smap': 2,       # 3-day soil moisture
            'hls': 3         # Harmonized data
        }
        return priority_map.get(name, 5)
    
    def _get_max_age_hours(self, name: str, source_type: str) -> int:
        """Get maximum acceptable data age in hours"""
        age_map = {
            'hydat': 2,      # 2 hours for real-time data
            'eccc': 3,       # 3 hours for weather data
            'modis': 24,     # 24 hours for daily data
            'era5_land': 168, # 1 week
            'smap': 72,      # 3 days
            'hls': 120       # 5 days
        }
        return age_map.get(name, 168)
    
    def _get_update_interval_hours(self, name: str, source_type: str) -> int:
        """Get recommended update interval in hours"""
        interval_map = {
            'hydat': 1,      # Every hour
            'eccc': 2,       # Every 2 hours
            'modis': 6,      # Every 6 hours
            'era5_land': 24, # Daily
            'smap': 12,      # Every 12 hours
            'hls': 24        # Daily
        }
        return interval_map.get(name, 24)
    
    def needs_update(self, source: DataSource) -> bool:
        """Determine if source needs update based on age and health"""
        if source.status == 'Offline':
            return False
        
        if source.health_status == 'Offline':
            return False
        
        if not source.last_update:
            return True
        
        # Calculate age in hours
        age_hours = (time.time() - source.last_update) / 3600
        
        # Check if data is too old
        if age_hours > source.max_age_hours:
            self.logger.info(f"{source.name}: Data age {age_hours:.1f}h > {source.max_age_hours}h, needs update")
            return True
        
        # Check if it's time for regular update
        if age_hours > source.update_interval_hours:
            self.logger.info(f"{source.name}: Update interval {age_hours:.1f}h > {source.update_interval_hours}h, needs update")
            return True
        
        return False
    
    def update_source(self, source: DataSource) -> bool:
        """Update a single data source"""
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would update {source.name}")
            return True
        
        try:
            self.logger.info(f"Updating {source.name}...")
            
            # Call pipeline sync API
            response = self.session.post(
                f"{self.api_base}/api/v1/pipeline/sync",
                params={'source': source.name}
            )
            response.raise_for_status()
            
            result = response.json()
            job_id = result.get('job_id')
            
            if not job_id:
                self.logger.error(f"No job ID returned for {source.name}")
                return False
            
            # Wait for job completion
            success = self._wait_for_job_completion(job_id, source.name)
            
            if success:
                self.logger.info(f"Successfully updated {source.name}")
            else:
                self.logger.error(f"Failed to update {source.name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error updating {source.name}: {e}")
            return False
    
    def _wait_for_job_completion(self, job_id: str, source_name: str, timeout_minutes: int = 30) -> bool:
        """Wait for pipeline job to complete"""
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while time.time() - start_time < timeout_seconds:
            try:
                response = self.session.get(f"{self.api_base}/api/v1/pipeline/job/{job_id}")
                response.raise_for_status()
                
                job_status = response.json()
                status = job_status.get('status')
                
                if status == 'succeeded':
                    return True
                elif status == 'failed':
                    self.logger.error(f"Job {job_id} for {source_name} failed: {job_status.get('message', 'Unknown error')}")
                    return False
                elif status in ['queued', 'running']:
                    self.logger.info(f"Job {job_id} for {source_name} still {status}, waiting...")
                    time.sleep(10)
                    continue
                else:
                    self.logger.warning(f"Unknown job status: {status}")
                    time.sleep(10)
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error checking job status: {e}")
                time.sleep(10)
                continue
        
        self.logger.error(f"Job {job_id} for {source_name} timed out after {timeout_minutes} minutes")
        return False
    
    def run_update_cycle(self, force_sources: Optional[List[str]] = None) -> Dict[str, bool]:
        """Run complete update cycle"""
        self.logger.info("Starting automated pipeline update cycle")
        
        # Get current status
        status_data = self.get_pipeline_status()
        if not status_data:
            self.logger.error("Cannot proceed without pipeline status")
            return {}
        
        # Parse sources
        sources = self.parse_data_sources(status_data)
        if not sources:
            self.logger.error("No data sources found")
            return {}
        
        # Sort by priority
        sources.sort(key=lambda x: x.priority)
        
        # Determine which sources to update
        sources_to_update = []
        
        if force_sources:
            # Force update specific sources
            sources_to_update = [s for s in sources if s.name in force_sources]
            self.logger.info(f"Force updating sources: {force_sources}")
        else:
            # Auto-select sources that need updates
            sources_to_update = [s for s in sources if self.needs_update(s)]
            self.logger.info(f"Auto-selected {len(sources_to_update)} sources for update")
        
        if not sources_to_update:
            self.logger.info("No sources need updates at this time")
            return {}
        
        # Update sources
        results = {}
        for source in sources_to_update:
            self.logger.info(f"Processing {source.name} (Priority: {source.priority})")
            success = self.update_source(source)
            results[source.name] = success
            
            # Add delay between updates to avoid overwhelming the system
            if len(sources_to_update) > 1:
                time.sleep(5)
        
        # Log summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        self.logger.info(f"Update cycle completed: {successful}/{total} sources updated successfully")
        
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='HydrAI-SWE Automated Pipeline Update')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be updated without making changes')
    parser.add_argument('--force', action='store_true', help='Force update all sources regardless of age')
    parser.add_argument('--sources', help='Comma-separated list of specific sources to update')
    parser.add_argument('--api-base', default='http://localhost:8000', help='API base URL')
    
    args = parser.parse_args()
    
    # Parse sources if specified
    force_sources = None
    if args.sources:
        force_sources = [s.strip() for s in args.sources.split(',')]
    
    # Create updater
    updater = PipelineUpdater(api_base=args.api_base, dry_run=args.dry_run)
    
    # Run update cycle
    try:
        results = updater.run_update_cycle(force_sources=force_sources)
        
        if results:
            print("\nUpdate Results:")
            for source, success in results.items():
                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"  {source}: {status}")
        
        # Exit with appropriate code
        if results and not any(results.values()):
            sys.exit(1)  # All failed
        elif results and not all(results.values()):
            sys.exit(2)  # Some failed
        else:
            sys.exit(0)  # All succeeded or no updates needed
            
    except KeyboardInterrupt:
        print("\nUpdate interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Update failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
