// HydrAI-SWE Main JavaScript
class DataComponent {
    constructor(element) {
        this.element = element;
        this.endpoint = element.dataset.endpoint;
        this.loadingElement = element.querySelector('.loading-indicator');
        this.dataElement = element.querySelector('.data-display');
        this.errorElement = element.querySelector('.error-display');
        this.timestampElement = element.querySelector('.timestamp');
        
        this.init();
    }

    async init() {
        await this.fetchData();
        this.setupAutoRefresh();
    }

    async fetchData() {
        this.showLoading();
        this.hideError();
        
        try {
            const data = await window.apiClient.get(this.endpoint);
            this.displayData(data);
            this.updateTimestamp();
        } catch (error) {
            this.showError(error.message);
        } finally {
            this.hideLoading();
        }
    }

    showLoading() {
        if (this.loadingElement) {
            this.loadingElement.style.display = 'flex';
        }
        if (this.dataElement) {
            this.dataElement.style.display = 'none';
        }
        if (this.errorElement) {
            this.errorElement.style.display = 'none';
        }
    }

    hideLoading() {
        if (this.loadingElement) {
            this.loadingElement.style.display = 'none';
        }
        if (this.dataElement) {
            this.dataElement.style.display = 'block';
        }
    }

    showError(message) {
        if (this.errorElement) {
            this.errorElement.querySelector('p').textContent = message;
            this.errorElement.style.display = 'flex';
        }
        if (this.dataElement) {
            this.dataElement.style.display = 'none';
        }
    }

    hideError() {
        if (this.errorElement) {
            this.errorElement.style.display = 'none';
        }
    }

    displayData(data) {
        // Display data based on endpoint type
        if (this.endpoint.includes('system-status')) {
            this.displaySystemStatus(data);
        } else if (this.endpoint.includes('data-quality')) {
            this.displayDataQuality(data);
        } else if (this.endpoint.includes('hydat-stations')) {
            this.displayHydatStations(data);
        } else if (this.endpoint.includes('prediction-status')) {
            this.displayPredictionStatus(data);
        } else if (this.endpoint.includes('flood-warning')) {
            this.displayFloodWarning(data);
        } else if (this.endpoint.includes('analysis/trends')) {
            this.displayTrendAnalysis(data);
        } else if (this.endpoint.includes('analysis/correlation')) {
            this.displayCorrelationAnalysis(data);
        } else if (this.endpoint.includes('forecast/7day')) {
            this.displayForecast(data);
        } else if (this.endpoint.includes('analysis/seasonal')) {
            this.displaySeasonalAnalysis(data);
        } else {
            console.log('Data received:', data);
        }
    }

    displaySystemStatus(data) {
        const uptimeElement = this.dataElement.querySelector('#system-uptime');
        if (uptimeElement && data.uptime) {
            uptimeElement.textContent = data.uptime;
            uptimeElement.className = 'metric success';
        }
    }

    displayDataQuality(data) {
        const qualityElement = this.dataElement.querySelector('#data-quality-score');
        const indicatorElement = this.dataElement.querySelector('#quality-indicator');
        
        if (qualityElement && data.overall_score) {
            qualityElement.textContent = data.overall_score + '%';
            qualityElement.className = 'metric success';
        }
        
        if (indicatorElement) {
            indicatorElement.className = 'status-indicator status-healthy';
        }
    }

    displayHydatStations(data) {
        const stationsElement = this.dataElement.querySelector('#active-stations');
        if (stationsElement && data.active_stations) {
            stationsElement.textContent = data.active_stations.toLocaleString();
            stationsElement.className = 'metric success';
        }
    }

    displayPredictionStatus(data) {
        const accuracyElement = this.dataElement.querySelector('#prediction-accuracy');
        if (accuracyElement && data.accuracy_nse) {
            accuracyElement.textContent = data.accuracy_nse;
            accuracyElement.className = 'metric success';
        }
    }

    displayFloodWarning(data) {
        const riskElement = this.dataElement.querySelector('#flood-risk-level');
        if (riskElement && data.risk_level) {
            riskElement.textContent = data.risk_level.toUpperCase();
            
            // Set color based on risk level
            if (data.risk_level === 'low') {
                riskElement.className = 'metric success';
            } else if (data.risk_level === 'medium') {
                riskElement.className = 'metric warning';
            } else {
                riskElement.className = 'metric danger';
            }
        }
    }

    displayTrendAnalysis(data) {
        const currentSwe = this.dataElement.querySelector('#current-swe');
        const averageSwe = this.dataElement.querySelector('#average-swe');
        const trendDirection = this.dataElement.querySelector('#trend-direction');
        const changePercentage = this.dataElement.querySelector('#change-percentage');
        
        if (currentSwe && data.current_swe_mm) {
            currentSwe.textContent = data.current_swe_mm;
        }
        if (averageSwe && data.average_swe_mm) {
            averageSwe.textContent = data.average_swe_mm;
        }
        if (trendDirection && data.trend) {
            trendDirection.textContent = data.trend;
            trendDirection.className = `trend-indicator ${data.trend}`;
        }
        if (changePercentage && data.change_percentage) {
            changePercentage.textContent = data.change_percentage;
            changePercentage.className = data.change_percentage < 0 ? 'metric-value negative' : 'metric-value positive';
        }
    }

    displayCorrelationAnalysis(data) {
        const correlationDisplay = this.dataElement.querySelector('#correlation-display');
        if (correlationDisplay && data.correlations) {
            let html = '<div class="correlation-list">';
            for (const [factor, correlation] of Object.entries(data.correlations)) {
                const strength = Math.abs(correlation) > 0.7 ? 'strong' : Math.abs(correlation) > 0.4 ? 'medium' : 'weak';
                const direction = correlation > 0 ? 'positive' : 'negative';
                html += `
                    <div class="correlation-item">
                        <span class="factor-name">${factor.replace('_', ' ').toUpperCase()}</span>
                        <div class="correlation-bar">
                            <div class="correlation-fill ${direction} ${strength}" style="width: ${Math.abs(correlation) * 100}%"></div>
                        </div>
                        <span class="correlation-value">${correlation.toFixed(2)}</span>
                    </div>
                `;
            }
            html += '</div>';
            correlationDisplay.innerHTML = html;
        }
    }

    displayForecast(data) {
        const forecastDisplay = this.dataElement.querySelector('#forecast-display');
        if (forecastDisplay && data.forecast_data) {
            let html = '<div class="forecast-chart">';
            data.forecast_data.forEach(day => {
                const confidence = Math.round(day.confidence * 100);
                html += `
                    <div class="forecast-day">
                        <div class="forecast-date">${day.date}</div>
                        <div class="forecast-value">
                            <span class="predicted">${day.predicted_swe}mm</span>
                            <span class="uncertainty">±${(day.uncertainty_upper - day.predicted_swe).toFixed(1)}mm</span>
                        </div>
                        <div class="forecast-confidence">
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${confidence}%"></div>
                            </div>
                            <span class="confidence-text">${confidence}%</span>
                        </div>
                        <div class="melt-rate">Melt: ${day.melt_rate}mm/day</div>
                    </div>
                `;
            });
            html += '</div>';
            forecastDisplay.innerHTML = html;
        }
    }

    displaySeasonalAnalysis(data) {
        const seasonalDisplay = this.dataElement.querySelector('#seasonal-display');
        if (seasonalDisplay && data.monthly_averages) {
            let html = '<div class="seasonal-chart">';
            const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
            
            months.forEach((month, index) => {
                const monthNum = index + 1;
                const avg = data.monthly_averages[monthNum] || 0;
                const std = data.monthly_std[monthNum] || 0;
                const height = (avg / Math.max(...Object.values(data.monthly_averages))) * 100;
                
                html += `
                    <div class="seasonal-month">
                        <div class="month-name">${month}</div>
                        <div class="month-bar">
                            <div class="month-fill" style="height: ${height}%"></div>
                            <div class="month-uncertainty" style="height: ${(std / Math.max(...Object.values(data.monthly_averages))) * 100}%"></div>
                        </div>
                        <div class="month-value">${avg.toFixed(1)}mm</div>
                    </div>
                `;
            });
            html += '</div>';
            seasonalDisplay.innerHTML = html;
        }
    }

    updateTimestamp() {
        if (this.timestampElement) {
            this.timestampElement.textContent = new Date().toLocaleString();
        }
    }

    setupAutoRefresh() {
        // Auto-refresh every 5 minutes
        setInterval(() => {
            this.fetchData();
        }, 5 * 60 * 1000);
    }
}

// Error Handler
class ErrorHandler {
    static handle(error, context = '') {
        console.error(`Error in ${context}:`, error);
        this.showUserError(this.getUserMessage(error));
    }
    
    static getUserMessage(error) {
        if (error.message.includes('Failed to fetch')) {
            return 'Unable to connect to the server. Please check your internet connection.';
        }
        
        if (error.message.includes('HTTP 500')) {
            return 'Server error occurred. Please try again later.';
        }
        
        if (error.message.includes('HTTP 404')) {
            return 'Requested data not found.';
        }
        
        return 'An unexpected error occurred. Please try again.';
    }
    
    static showUserError(message) {
        const notification = document.createElement('div');
        notification.className = 'error-notification';
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #e74c3c;
            color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            z-index: 1000;
            max-width: 300px;
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize data components
    document.querySelectorAll('.data-card').forEach(element => {
        new DataComponent(element);
    });
    
    // Global error handling
    window.addEventListener('error', (event) => {
        ErrorHandler.handle(event.error, 'Global');
    });
    
    window.addEventListener('unhandledrejection', (event) => {
        ErrorHandler.handle(event.reason, 'Promise rejection');
    });
    
    console.log('HydrAI-SWE Frontend initialized');
});
