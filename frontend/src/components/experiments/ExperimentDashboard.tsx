/**
 * A/B Testing Experiment Dashboard Component
 * Provides real-time monitoring, results visualization, and experiment management
 */

import React, { useState, useEffect, useCallback } from 'react';
import { 
  Play, 
  Pause, 
  Square, 
  BarChart3, 
  AlertTriangle,
  CheckCircle,
  Clock,
  Target,
  Zap
} from 'lucide-react';

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

interface ExperimentVariant {
  id: string;
  name: string;
  description?: string;
  traffic_percentage: number;
  is_control: boolean;
}

interface MetricDefinition {
  id: string;
  name: string;
  type: string;
  is_primary: boolean;
  target_value?: number;
}

interface StoppingRule {
  type: string;
  threshold: number;
  enabled: boolean;
}

interface Experiment {
  id: string;
  name: string;
  description?: string;
  hypothesis: string;
  status: 'draft' | 'running' | 'paused' | 'completed' | 'stopped';
  variants: ExperimentVariant[];
  metrics: MetricDefinition[];
  stopping_rules?: StoppingRule[];
  target_sample_size?: number;
  current_sample_size: number;
  confidence_level: number;
  created_at: string;
  started_at?: string;
  ended_at?: string;
  statistical_power?: number;
  significance_reached: boolean;
}

interface VariantResults {
  variant_id: string;
  variant_name: string;
  is_control: boolean;
  participant_count: number;
  metrics: Record<string, {
    count: number;
    mean: number;
    std: number;
    min: number;
    max: number;
  }>;
}

interface StatisticalTest {
  metric: string;
  control_variant: string;
  treatment_variant: string;
  test_type: string;
  p_value: number;
  confidence_interval: [number, number];
  effect_size: number;
  significant: boolean;
  control_mean: number;
  treatment_mean: number;
}

interface ExperimentResults {
  experiment_id: string;
  experiment_name: string;
  total_participants: number;
  variants: VariantResults[];
  statistical_tests: Record<string, StatisticalTest>;
}

// ============================================================================
// DASHBOARD COMPONENT
// ============================================================================

const ExperimentDashboard: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<Experiment | null>(null);
  const [experimentResults, setExperimentResults] = useState<ExperimentResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshInterval, setRefreshInterval] = useState<NodeJS.Timeout | null>(null);

  // ============================================================================
  // API FUNCTIONS
  // ============================================================================

  const fetchExperiments = useCallback(async () => {
    try {
      const response = await fetch('/api/experiments/experiments', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch experiments');
      }
      
      const data = await response.json();
      setExperiments(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, []);

  const fetchExperimentResults = useCallback(async (experimentId: string) => {
    try {
      const response = await fetch(`/api/experiments/experiments/${experimentId}/results`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch results');
      }
      
      const data = await response.json();
      setExperimentResults(data);
    } catch (err) {
      console.error('Error fetching results:', err);
    }
  }, []);

  const startExperiment = async (experimentId: string) => {
    try {
      const response = await fetch(`/api/experiments/experiments/${experimentId}/start`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      
      if (!response.ok) {
        throw new Error('Failed to start experiment');
      }
      
      await fetchExperiments();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start experiment');
    }
  };

  const stopExperiment = async (experimentId: string) => {
    try {
      const response = await fetch(`/api/experiments/experiments/${experimentId}/stop`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      
      if (!response.ok) {
        throw new Error('Failed to stop experiment');
      }
      
      await fetchExperiments();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to stop experiment');
    }
  };

  // ============================================================================
  // EFFECTS
  // ============================================================================

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await fetchExperiments();
      setLoading(false);
    };
    
    loadData();
  }, [fetchExperiments]);

  useEffect(() => {
    if (selectedExperiment && selectedExperiment.status === 'running') {
      fetchExperimentResults(selectedExperiment.id);
      
      // Set up auto-refresh for running experiments
      const interval = setInterval(() => {
        fetchExperimentResults(selectedExperiment.id);
        fetchExperiments();
      }, 30000); // Refresh every 30 seconds
      
      setRefreshInterval(interval);
      
      return () => {
        if (interval) clearInterval(interval);
      };
    } else {
      if (refreshInterval) {
        clearInterval(refreshInterval);
        setRefreshInterval(null);
      }
    }
  }, [selectedExperiment, fetchExperimentResults, fetchExperiments, refreshInterval]);

  // ============================================================================
  // UTILITY FUNCTIONS
  // ============================================================================

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'text-green-600 bg-green-100';
      case 'draft': return 'text-gray-600 bg-gray-100';
      case 'paused': return 'text-yellow-600 bg-yellow-100';
      case 'completed': return 'text-blue-600 bg-blue-100';
      case 'stopped': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <Play className="w-4 h-4" />;
      case 'draft': return <Clock className="w-4 h-4" />;
      case 'paused': return <Pause className="w-4 h-4" />;
      case 'completed': return <CheckCircle className="w-4 h-4" />;
      case 'stopped': return <Square className="w-4 h-4" />;
      default: return <Clock className="w-4 h-4" />;
    }
  };

  const formatPercentage = (value: number) => `${(value * 100).toFixed(2)}%`;
  const formatDate = (dateString: string) => new Date(dateString).toLocaleDateString();

  // ============================================================================
  // RENDER FUNCTIONS
  // ============================================================================

  const renderExperimentCard = (experiment: Experiment) => (
    <div
      key={experiment.id}
      className={`p-6 border rounded-lg cursor-pointer transition-all hover:shadow-md ${
        selectedExperiment?.id === experiment.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
      }`}
      onClick={() => setSelectedExperiment(experiment)}
    >
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">{experiment.name}</h3>
          <p className="text-sm text-gray-600 mt-1">{experiment.description}</p>
        </div>
        <div className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(experiment.status)}`}>
          {getStatusIcon(experiment.status)}
          {experiment.status.charAt(0).toUpperCase() + experiment.status.slice(1)}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-600">{experiment.current_sample_size}</div>
          <div className="text-xs text-gray-500">Participants</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-green-600">{experiment.variants.length}</div>
          <div className="text-xs text-gray-500">Variants</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-purple-600">{experiment.metrics.length}</div>
          <div className="text-xs text-gray-500">Metrics</div>
        </div>
      </div>

      <div className="flex justify-between items-center">
        <span className="text-sm text-gray-500">
          Created {formatDate(experiment.created_at)}
        </span>
        {experiment.significance_reached && (
          <div className="flex items-center gap-1 text-green-600">
            <Target className="w-4 h-4" />
            <span className="text-xs">Significant</span>
          </div>
        )}
      </div>
    </div>
  );

  const renderVariantResults = (variant: VariantResults) => (
    <div key={variant.variant_id} className="p-4 border rounded-lg">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-medium text-gray-900">{variant.variant_name}</h4>
        <div className="flex items-center gap-2">
          {variant.is_control && (
            <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
              Control
            </span>
          )}
          <span className="text-sm text-gray-600">{variant.participant_count} users</span>
        </div>
      </div>
      
      <div className="space-y-2">
        {Object.entries(variant.metrics).map(([metricId, stats]) => (
          <div key={metricId} className="flex justify-between">
            <span className="text-sm text-gray-600">{metricId}:</span>
            <span className="text-sm font-medium">{stats.mean.toFixed(3)} ±{stats.std.toFixed(3)}</span>
          </div>
        ))}
      </div>
    </div>
  );

  const renderStatisticalTests = () => {
    if (!experimentResults?.statistical_tests) return null;

    return (
      <div className="space-y-4">
        {Object.entries(experimentResults.statistical_tests).map(([testKey, test]) => (
          <div key={testKey} className="p-4 border rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-medium text-gray-900">{test.metric} Analysis</h4>
              <div className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs ${
                test.significant ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
              }`}>
                {test.significant ? <CheckCircle className="w-3 h-3" /> : <AlertTriangle className="w-3 h-3" />}
                {test.significant ? 'Significant' : 'Not Significant'}
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-600">P-value:</span>
                <span className="ml-2 font-medium">{test.p_value.toFixed(4)}</span>
              </div>
              <div>
                <span className="text-gray-600">Effect Size:</span>
                <span className="ml-2 font-medium">{test.effect_size.toFixed(3)}</span>
              </div>
              <div>
                <span className="text-gray-600">Control Mean:</span>
                <span className="ml-2 font-medium">{test.control_mean.toFixed(3)}</span>
              </div>
              <div>
                <span className="text-gray-600">Treatment Mean:</span>
                <span className="ml-2 font-medium">{test.treatment_mean.toFixed(3)}</span>
              </div>
            </div>
            
            <div className="mt-2 text-sm">
              <span className="text-gray-600">95% CI:</span>
              <span className="ml-2 font-medium">
                [{test.confidence_interval[0].toFixed(3)}, {test.confidence_interval[1].toFixed(3)}]
              </span>
            </div>
          </div>
        ))}
      </div>
    );
  };

  // ============================================================================
  // MAIN RENDER
  // ============================================================================

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
        <div className="flex items-center gap-2 text-red-800">
          <AlertTriangle className="w-5 h-5" />
          <span>Error: {error}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">A/B Testing Dashboard</h1>
        <p className="text-gray-600">Monitor and analyze your experiments in real-time</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Experiments List */}
        <div className="lg:col-span-1 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-gray-900">Experiments</h2>
            <button
              onClick={fetchExperiments}
              className="p-2 text-gray-500 hover:text-gray-700"
              title="Refresh"
            >
              <Zap className="w-4 h-4" />
            </button>
          </div>
          
          {experiments.length === 0 ? (
            <div className="p-6 text-center text-gray-500 border-2 border-dashed border-gray-300 rounded-lg">
              No experiments found. Create your first experiment to get started.
            </div>
          ) : (
            <div className="space-y-3">
              {experiments.map(renderExperimentCard)}
            </div>
          )}
        </div>

        {/* Experiment Details */}
        <div className="lg:col-span-2">
          {selectedExperiment ? (
            <div className="space-y-6">
              {/* Header */}
              <div className="p-6 bg-white border rounded-lg">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-2xl font-bold text-gray-900">{selectedExperiment.name}</h2>
                  <div className="flex gap-2">
                    {selectedExperiment.status === 'draft' && (
                      <button
                        onClick={() => startExperiment(selectedExperiment.id)}
                        className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
                      >
                        <Play className="w-4 h-4" />
                        Start
                      </button>
                    )}
                    {selectedExperiment.status === 'running' && (
                      <button
                        onClick={() => stopExperiment(selectedExperiment.id)}
                        className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
                      >
                        <Square className="w-4 h-4" />
                        Stop
                      </button>
                    )}
                  </div>
                </div>
                
                <p className="text-gray-600 mb-4">{selectedExperiment.hypothesis}</p>
                
                <div className="grid grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-lg font-bold text-blue-600">{selectedExperiment.current_sample_size}</div>
                    <div className="text-sm text-gray-500">Current Sample</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold text-green-600">
                      {selectedExperiment.target_sample_size || 'N/A'}
                    </div>
                    <div className="text-sm text-gray-500">Target Sample</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold text-purple-600">
                      {formatPercentage(selectedExperiment.confidence_level)}
                    </div>
                    <div className="text-sm text-gray-500">Confidence</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold text-orange-600">
                      {selectedExperiment.statistical_power ? 
                        formatPercentage(selectedExperiment.statistical_power) : 'N/A'}
                    </div>
                    <div className="text-sm text-gray-500">Power</div>
                  </div>
                </div>
              </div>

              {/* Variants */}
              <div className="p-6 bg-white border rounded-lg">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Variants</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {selectedExperiment.variants.map(variant => (
                    <div key={variant.id} className="p-4 border rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium">{variant.name}</h4>
                        {variant.is_control && (
                          <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                            Control
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-600 mb-2">{variant.description}</p>
                      <div className="text-sm">
                        <span className="text-gray-500">Traffic:</span>
                        <span className="ml-2 font-medium">{formatPercentage(variant.traffic_percentage / 100)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Results */}
              {experimentResults && (
                <>
                  <div className="p-6 bg-white border rounded-lg">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Variant Performance</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {experimentResults.variants.map(renderVariantResults)}
                    </div>
                  </div>

                  <div className="p-6 bg-white border rounded-lg">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Statistical Analysis</h3>
                    {renderStatisticalTests()}
                  </div>
                </>
              )}
            </div>
          ) : (
            <div className="p-12 text-center text-gray-500 border-2 border-dashed border-gray-300 rounded-lg">
              <BarChart3 className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              <h3 className="text-lg font-medium mb-2">Select an Experiment</h3>
              <p>Choose an experiment from the list to view detailed analytics and results.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ExperimentDashboard; 