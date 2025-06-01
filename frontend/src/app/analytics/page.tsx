'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  Users, 
  Activity, 
  DollarSign, 
  CheckCircle2, 
  RefreshCw
} from 'lucide-react';

// Import analytics API functions
import { analyticsApi } from '../lib/api/analytics';
import { RealTimeUpdate } from '../types/analytics';

// Import UI components
import { MetricCard } from '../components/analytics/MetricCard';
import { TrendChart } from '../components/analytics/TrendChart';
import { EngagementChart } from '../components/analytics/EngagementChart';
import { AgreementChart } from '../components/analytics/AgreementChart';
import { ModelComparisonTable } from '../components/analytics/ModelComparisonTable';
import { SystemHealthIndicator } from '../components/analytics/SystemHealthIndicator';
import { RealTimeUpdates } from '../components/analytics/RealTimeUpdates';

type TimeRange = '1h' | '24h' | '7d' | '30d' | '90d' | '365d';

export default function AnalyticsDashboard() {
  const [timeRange, setTimeRange] = useState<TimeRange>('24h');
  const [isRealTimeEnabled, setIsRealTimeEnabled] = useState(true);

  // Fetch system overview data
  const { 
    data: systemOverview, 
    isLoading: overviewLoading, 
    refetch: refetchOverview 
  } = useQuery({
    queryKey: ['systemOverview', timeRange],
    queryFn: () => analyticsApi.getSystemOverview(timeRange),
    refetchInterval: isRealTimeEnabled ? 30000 : false, // Refresh every 30s if real-time enabled
  });

  // Fetch user engagement data
  const { 
    data: userEngagement, 
    isLoading: engagementLoading 
  } = useQuery({
    queryKey: ['userEngagement', timeRange],
    queryFn: () => analyticsApi.getUserEngagement(parseInt(timeRange.replace(/[^\d]/g, '')) || 24),
    refetchInterval: isRealTimeEnabled ? 60000 : false, // Refresh every 60s
  });

  // Fetch LLM-human agreement data
  const { 
    data: agreementAnalysis, 
    isLoading: agreementLoading 
  } = useQuery({
    queryKey: ['agreementAnalysis', timeRange],
    queryFn: () => analyticsApi.getLLMHumanAgreement(parseInt(timeRange.replace(/[^\d]/g, '')) || 24),
    refetchInterval: isRealTimeEnabled ? 60000 : false,
  });

  // Fetch acceptance rate data
  const { 
    data: acceptanceRates, 
    isLoading: acceptanceLoading 
  } = useQuery({
    queryKey: ['acceptanceRates', timeRange],
    queryFn: () => analyticsApi.getAcceptanceRates(parseInt(timeRange.replace(/[^\d]/g, '')) || 24),
    refetchInterval: isRealTimeEnabled ? 60000 : false,
  });

  // Fetch model comparison data
  const { 
    data: modelComparison, 
    isLoading: comparisonLoading 
  } = useQuery({
    queryKey: ['modelComparison', timeRange],
    queryFn: () => analyticsApi.getModelComparison(timeRange),
    refetchInterval: isRealTimeEnabled ? 120000 : false, // Refresh every 2 minutes
  });

  // Fetch system health data
  const { 
    data: systemHealth, 
    isLoading: healthLoading 
  } = useQuery({
    queryKey: ['systemHealth'],
    queryFn: analyticsApi.getSystemHealth,
    refetchInterval: isRealTimeEnabled ? 10000 : false, // Refresh every 10s
  });

  const timeRangeOptions = [
    { value: '1h', label: 'Last Hour' },
    { value: '24h', label: 'Last 24 Hours' },
    { value: '7d', label: 'Last 7 Days' },
    { value: '30d', label: 'Last 30 Days' },
    { value: '90d', label: 'Last 90 Days' },
    { value: '365d', label: 'Last Year' },
  ];

  const isLoading = overviewLoading || engagementLoading || agreementLoading || acceptanceLoading;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Analytics Dashboard</h1>
              <p className="mt-1 text-sm text-gray-500">
                Platform performance, user engagement, and AI evaluation insights
              </p>
            </div>
            
            <div className="flex items-center gap-4">
              {/* Real-time toggle */}
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={isRealTimeEnabled}
                  onChange={(e) => setIsRealTimeEnabled(e.target.checked)}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm text-gray-600">Real-time updates</span>
              </label>

              {/* Time range selector */}
              <select
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value as TimeRange)}
                className="rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                {timeRangeOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>

              {/* Refresh button */}
              <button
                onClick={() => refetchOverview()}
                disabled={isLoading}
                className="inline-flex items-center gap-2 px-3 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
              >
                <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                Refresh
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* System Health Indicator */}
        {systemHealth && (
          <SystemHealthIndicator 
            health={systemHealth} 
            loading={healthLoading}
            className="mb-6"
          />
        )}

        {/* Key Metrics Overview */}
        {systemOverview && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <MetricCard
              title="Total Evaluations"
              value={systemOverview.evaluation_metrics.total_evaluations}
              change={systemOverview.evaluation_metrics.throughput_per_hour}
              changeLabel="per hour"
              icon={<Activity className="w-5 h-5" />}
              trend="up"
            />
            
            <MetricCard
              title="Success Rate"
              value={`${(systemOverview.evaluation_metrics.success_rate * 100).toFixed(1)}%`}
              change={systemOverview.evaluation_metrics.success_rate}
              changeLabel="accuracy"
              icon={<CheckCircle2 className="w-5 h-5" />}
              trend={systemOverview.evaluation_metrics.success_rate > 0.9 ? "up" : "down"}
            />
            
            <MetricCard
              title="Total Cost"
              value={`$${systemOverview.cost_metrics.total_cost_usd.toFixed(2)}`}
              change={systemOverview.cost_metrics.cost_per_evaluation}
              changeLabel="per eval"
              icon={<DollarSign className="w-5 h-5" />}
              trend="neutral"
            />
            
            <MetricCard
              title="Active Users"
              value={userEngagement?.user_metrics.active_users || 0}
              change={userEngagement?.user_metrics.new_users || 0}
              changeLabel="new users"
              icon={<Users className="w-5 h-5" />}
              trend="up"
            />
          </div>
        )}

        {/* Main Analytics Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Performance Trends Chart */}
          {systemOverview && (
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Trends</h3>
              <TrendChart 
                data={systemOverview.performance_trends}
                loading={overviewLoading}
              />
            </div>
          )}

          {/* User Engagement Chart */}
          {userEngagement && (
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">User Engagement</h3>
              <EngagementChart 
                data={userEngagement}
                loading={engagementLoading}
              />
            </div>
          )}
        </div>

        {/* AI-Human Agreement Analysis */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Agreement Analysis */}
          {agreementAnalysis && (
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">LLM ↔ Human Agreement</h3>
              <AgreementChart 
                data={agreementAnalysis}
                loading={agreementLoading}
              />
            </div>
          )}

          {/* Acceptance Rates */}
          {acceptanceRates && (
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">AI Suggestion Acceptance</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Overall Acceptance Rate</span>
                  <span className="text-lg font-semibold text-gray-900">
                    {(acceptanceRates.acceptance_rate * 100).toFixed(1)}%
                  </span>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">High Confidence</span>
                    <span className="font-medium">
                      {(acceptanceRates.acceptance_by_confidence.high * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Medium Confidence</span>
                    <span className="font-medium">
                      {(acceptanceRates.acceptance_by_confidence.medium * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Low Confidence</span>
                    <span className="font-medium">
                      {(acceptanceRates.acceptance_by_confidence.low * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Model Performance Comparison */}
        {modelComparison && (
          <div className="bg-white p-6 rounded-lg border border-gray-200 mb-8">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Performance Comparison</h3>
            <ModelComparisonTable 
              data={modelComparison}
              loading={comparisonLoading}
            />
          </div>
        )}

        {/* Real-time Updates Component */}
        {isRealTimeEnabled && (
          <RealTimeUpdates 
            onUpdate={(update: RealTimeUpdate) => {
              // Handle real-time updates
              console.log('Real-time update:', update);
            }}
          />
        )}
      </div>
    </div>
  );
} 