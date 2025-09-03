// Shared Dashboard JavaScript Functions for All Language Pages
// This file contains the core functionality that was fixed in enhanced_en.html

// API Configuration
const API_BASE_URL = '/api/swe';
const API_FLOOD_URL = '/api/v1/flood';
const API_AGRICULTURE_URL = '/api/v1/agriculture';
const API_TIMEOUT = 30000;
const MAX_RETRIES = 3;

// Generic API fetch with error handling and retry mechanism
async function apiRequest(endpoint, options = {}, baseUrl = API_BASE_URL) {
    let lastError;
    
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT);
            
            const response = await fetch(`${baseUrl}${endpoint}`, {
                ...options,
                signal: controller.signal,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            lastError = error;
            if (attempt < MAX_RETRIES && !error.name?.includes('Abort')) {
                await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
                continue;
            }
            break;
        }
    }
    
    throw lastError;
}

// Fetch functions
async function fetchCurrentSeasonSummary() {
    return await apiRequest('/current-season-summary');
}

async function fetchFloodRiskAssessment() {
    return await apiRequest('/flood-risk');
}

async function fetchRegionalForecastDetails() {
    return await apiRequest('/regional-forecast');
}

// Notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 10000;
        max-width: 300px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    `;
    
    const colors = {
        'success': '#27ae60',
        'error': '#e74c3c',
        'warning': '#f39c12',
        'info': '#3498db'
    };
    
    notification.style.backgroundColor = colors[type] || colors.info;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 3000);
}

// Load dashboard data functions
async function loadCurrentSeasonSummary() {
    try {
        console.log('Loading current season summary...');
        const data = await fetchCurrentSeasonSummary();
        
        const totalSnowEl = document.getElementById('total-snow-value');
        const vsHistoricalEl = document.getElementById('vs-historical-value');
        const peakDateEl = document.getElementById('peak-date-value');
        const activeStationsEl = document.getElementById('active-stations-value');
        
        if (totalSnowEl && data.total_snow) totalSnowEl.textContent = data.total_snow.value;
        if (vsHistoricalEl && data.vs_historical) vsHistoricalEl.textContent = data.vs_historical.value;
        if (peakDateEl && data.peak_date) peakDateEl.textContent = data.peak_date.value;
        if (activeStationsEl && data.active_stations) activeStationsEl.textContent = data.active_stations.value;
        
    } catch (error) {
        console.error('Failed to load current season summary:', error);
        const elements = ['total-snow-value', 'vs-historical-value', 'peak-date-value', 'active-stations-value'];
        elements.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = 'Data unavailable';
        });
    }
}

async function loadFloodRiskAssessment() {
    try {
        const data = await fetchFloodRiskAssessment();
        
        const riskLevelEl = document.getElementById('risk-level-value');
        const peakRiskEl = document.getElementById('peak-risk-period-value');
        const regionsAtRiskEl = document.getElementById('regions-at-risk-value');
        const alertLeadTimeEl = document.getElementById('alert-lead-time-value');
        const floodAlertEl = document.getElementById('flood-alert-content');
        
        if (riskLevelEl && data.risk_level) {
            riskLevelEl.textContent = data.risk_level.value;
            if (data.risk_level.color) riskLevelEl.style.color = data.risk_level.color;
        }
        if (peakRiskEl && data.peak_risk_period) peakRiskEl.textContent = data.peak_risk_period.value;
        if (regionsAtRiskEl && data.regions_at_risk) regionsAtRiskEl.textContent = data.regions_at_risk.value;
        if (alertLeadTimeEl && data.alert_lead_time) alertLeadTimeEl.textContent = data.alert_lead_time.value;
        if (floodAlertEl && data.alert_message) floodAlertEl.textContent = data.alert_message;
        
    } catch (error) {
        console.error('Failed to load flood risk assessment:', error);
        const elements = ['risk-level-value', 'peak-risk-period-value', 'regions-at-risk-value', 'alert-lead-time-value'];
        elements.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = 'Data unavailable';
        });
        
        const floodAlertEl = document.getElementById('flood-alert-content');
        if (floodAlertEl) floodAlertEl.textContent = 'Flood risk assessment unavailable';
    }
}

async function loadRegionalForecastDetails() {
    try {
        const data = await fetchRegionalForecastDetails();
        const tableBody = document.getElementById('regional-forecast-table');
        
        if (!tableBody) return;
        
        tableBody.innerHTML = '';
        
        if (data && data.regions && Array.isArray(data.regions)) {
            data.regions.forEach(region => {
                const row = document.createElement('tr');
                
                const riskClass = region.risk_level && region.risk_level.includes('High') ? 'style="color: #e74c3c; font-weight: bold;"' :
                                 region.risk_level && region.risk_level.includes('Moderate') ? 'style="color: #f39c12; font-weight: bold;"' :
                                 region.risk_level && region.risk_level.includes('Low') ? 'style="color: #27ae60; font-weight: bold;"' :
                                 'style="color: #95a5a6; font-weight: bold;"';
            
                const forecastClass = region.forecast_7day && region.forecast_7day.startsWith('+') ? 'style="color: #3498db;"' :
                                     region.forecast_7day && region.forecast_7day.startsWith('-') ? 'style="color: #e67e22;"' :
                                     'style="color: #95a5a6;"';
            
                row.innerHTML = `
                    <td>${region.name || 'N/A'}</td>
                    <td>${region.current_swe || 'N/A'}</td>
                    <td ${forecastClass}>${region.forecast_7day || 'N/A'}</td>
                    <td>${region.peak_runoff_date || 'N/A'}</td>
                    <td>${region.expected_volume || 'N/A'}</td>
                    <td ${riskClass}>${region.risk_level || 'N/A'}</td>
                `;
                
                tableBody.appendChild(row);
            });
        } else {
            tableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #666;">No regional forecast data available</td></tr>';
        }
        
    } catch (error) {
        console.error('Failed to load regional forecast details:', error);
        const tableBody = document.getElementById('regional-forecast-table');
        if (tableBody) {
            tableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #e74c3c;">Failed to load regional forecast data</td></tr>';
        }
    }
}

async function initializeAllData() {
    try {
        console.log('Starting to initialize all data...');
        
        await loadCurrentSeasonSummary();
        await loadFloodRiskAssessment();
        await loadRegionalForecastDetails();
        
        console.log('All data components loaded successfully');
        showNotification('Dashboard data loaded successfully', 'success');
        
    } catch (error) {
        console.error('Failed to load all data:', error);
        console.log('Loading fallback data due to API issues...');
        await loadFallbackData();
        showNotification('Using fallback data due to API connection issues', 'warning');
    }
}

async function loadFallbackData() {
    try {
        console.log('Loading fallback data...');
        
        const totalSnowEl = document.getElementById('total-snow-value');
        const vsHistoricalEl = document.getElementById('vs-historical-value');
        const peakDateEl = document.getElementById('peak-date-value');
        const activeStationsEl = document.getElementById('active-stations-value');
        
        // 显示数据不可用，不使用硬编码数据
        if (totalSnowEl) totalSnowEl.textContent = 'N/A';
        if (vsHistoricalEl) vsHistoricalEl.textContent = 'N/A';
        if (peakDateEl) peakDateEl.textContent = 'N/A';
        if (activeStationsEl) activeStationsEl.textContent = 'N/A';
        
        const riskLevelEl = document.getElementById('risk-level-value');
        const peakRiskEl = document.getElementById('peak-risk-period-value');
        const regionsAtRiskEl = document.getElementById('regions-at-risk-value');
        const alertLeadTimeEl = document.getElementById('alert-lead-time-value');
        const floodAlertEl = document.getElementById('flood-alert-content');
        
        if (riskLevelEl) riskLevelEl.textContent = 'N/A';
        if (peakRiskEl) peakRiskEl.textContent = 'N/A';
        if (regionsAtRiskEl) regionsAtRiskEl.textContent = 'N/A';
        if (alertLeadTimeEl) alertLeadTimeEl.textContent = 'N/A';
        if (floodAlertEl) floodAlertEl.textContent = 'Data not available. Please check data pipeline connection.';
        
        console.log('Data unavailable - no fallback data allowed');
        
    } catch (error) {
        console.error('Failed to load fallback data:', error);
        showNotification('Failed to load any data. Please refresh the page.', 'error');
    }
}

// Language switching functionality
function switchLanguage(lang) {
    localStorage.setItem('selectedLanguage', lang);
    
    if (lang === 'fr') {
        window.location.href = '/ui/francais';
    } else if (lang === 'cr') {
        window.location.href = '/ui/cree';
    } else {
        window.location.href = '/ui/enhanced_en';
    }
}

// Initialize language system
function initializeLanguageSystem() {
    const savedLanguage = localStorage.getItem('selectedLanguage') || 'en';
    const languageSelect = document.getElementById('languageSelect');
    if (languageSelect) {
        languageSelect.value = savedLanguage;
    }
}

// Initialize dashboard data loading
function initializeDashboard() {
    showNotification('Loading dashboard data...', 'info');
    
    setTimeout(async () => {
        try {
            await initializeAllData();
        } catch (error) {
            console.error('Dashboard initialization failed:', error);
        }
    }, 200);
}
