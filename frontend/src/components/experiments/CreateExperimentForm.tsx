/**
 * Create Experiment Form Component
 * Comprehensive form for setting up new A/B testing experiments
 */

import React, { useState } from 'react';
import { 
  Plus, 
  Minus, 
  Calculator, 
  Target, 
  AlertCircle,
  Info
} from 'lucide-react';

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

interface ExperimentVariant {
  id: string;
  name: string;
  description: string;
  traffic_percentage: number;
  is_control: boolean;
  model_config_override?: Record<string, string | number | boolean>;
  prompt_template_override?: string;
}

interface MetricDefinition {
  id: string;
  name: string;
  type: 'conversion_rate' | 'satisfaction_score' | 'completion_rate' | 'error_rate' | 'response_time' | 'custom';
  description: string;
  aggregation_method: 'mean' | 'sum' | 'count';
  is_primary: boolean;
  target_value?: number;
}

interface StoppingRule {
  type: 'statistical_significance' | 'minimum_effect_size' | 'time_based' | 'sample_size';
  threshold: number;
  enabled: boolean;
  description: string;
}

interface SegmentationCriteria {
  type: 'random' | 'cohort' | 'attribute' | 'custom';
  criteria: Record<string, string | number | boolean>;
  percentage?: number;
}

interface ExperimentFormData {
  name: string;
  description: string;
  hypothesis: string;
  variants: ExperimentVariant[];
  metrics: MetricDefinition[];
  segmentation?: SegmentationCriteria;
  stopping_rules: StoppingRule[];
  target_sample_size?: number;
  estimated_duration_days?: number;
  confidence_level: number;
  minimum_effect_size?: number;
}

interface SampleSizeCalculation {
  sample_size_per_variant: number;
  total_sample_size: number;
  parameters: {
    baseline_conversion_rate: number;
    minimum_effect_size: number;
    confidence_level: number;
    statistical_power: number;
    two_tailed: boolean;
  };
}

// ============================================================================
// FORM COMPONENT
// ============================================================================

const CreateExperimentForm: React.FC<{ onClose: () => void; onSuccess: () => void }> = ({ 
  onClose, 
  onSuccess 
}) => {
  const [formData, setFormData] = useState<ExperimentFormData>({
    name: '',
    description: '',
    hypothesis: '',
    variants: [
      {
        id: 'control',
        name: 'Control',
        description: 'Original version',
        traffic_percentage: 50,
        is_control: true,
      },
      {
        id: 'treatment',
        name: 'Treatment',
        description: 'New version',
        traffic_percentage: 50,
        is_control: false,
      },
    ],
    metrics: [
      {
        id: 'conversion_rate',
        name: 'Conversion Rate',
        type: 'conversion_rate',
        description: 'Primary conversion metric',
        aggregation_method: 'mean',
        is_primary: true,
      },
    ],
    stopping_rules: [
      {
        type: 'statistical_significance',
        threshold: 0.05,
        enabled: true,
        description: 'Stop when p-value < 0.05',
      },
    ],
    confidence_level: 0.95,
  });

  const [sampleSizeCalc, setSampleSizeCalc] = useState({
    baseline_conversion_rate: 0.1,
    minimum_effect_size: 0.02,
    confidence_level: 0.95,
    statistical_power: 0.8,
    two_tailed: true,
  });

  const [calculatedSampleSize, setCalculatedSampleSize] = useState<SampleSizeCalculation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState(1);

  // ============================================================================
  // UTILITY FUNCTIONS
  // ============================================================================

  const validateForm = (): string | null => {
    if (!formData.name.trim()) return 'Experiment name is required';
    if (!formData.hypothesis.trim()) return 'Hypothesis is required';
    
    const totalTraffic = formData.variants.reduce((sum, v) => sum + v.traffic_percentage, 0);
    if (Math.abs(totalTraffic - 100) > 0.01) return 'Variant traffic percentages must sum to 100%';
    
    const controlCount = formData.variants.filter(v => v.is_control).length;
    if (controlCount !== 1) return 'Exactly one variant must be marked as control';
    
    if (formData.metrics.length === 0) return 'At least one metric is required';
    
    const primaryMetrics = formData.metrics.filter(m => m.is_primary).length;
    if (primaryMetrics === 0) return 'At least one primary metric is required';
    
    return null;
  };

  // ============================================================================
  // API FUNCTIONS
  // ============================================================================

  const calculateSampleSize = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/experiments/sample-size', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify(sampleSizeCalc),
      });

      if (!response.ok) {
        throw new Error('Failed to calculate sample size');
      }

      const result = await response.json();
      setCalculatedSampleSize(result);
      setFormData(prev => ({ ...prev, target_sample_size: result.total_sample_size }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sample size calculation failed');
    } finally {
      setLoading(false);
    }
  };

  const createExperiment = async () => {
    const validationError = validateForm();
    if (validationError) {
      setError(validationError);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/experiments/experiments', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Failed to create experiment');
      }

      onSuccess();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create experiment');
    } finally {
      setLoading(false);
    }
  };

  // ============================================================================
  // FORM HANDLERS
  // ============================================================================

  const addVariant = () => {
    const newVariant: ExperimentVariant = {
      id: `variant_${Date.now()}`,
      name: `Variant ${formData.variants.length + 1}`,
      description: '',
      traffic_percentage: 0,
      is_control: false,
    };
    
    setFormData(prev => ({
      ...prev,
      variants: [...prev.variants, newVariant],
    }));
  };

  const updateVariant = (index: number, updates: Partial<ExperimentVariant>) => {
    setFormData(prev => ({
      ...prev,
      variants: prev.variants.map((variant, i) => 
        i === index ? { ...variant, ...updates } : variant
      ),
    }));
  };

  const removeVariant = (index: number) => {
    setFormData(prev => ({
      ...prev,
      variants: prev.variants.filter((_, i) => i !== index),
    }));
  };

  const addMetric = () => {
    const newMetric: MetricDefinition = {
      id: `metric_${Date.now()}`,
      name: '',
      type: 'custom',
      description: '',
      aggregation_method: 'mean',
      is_primary: false,
    };
    
    setFormData(prev => ({
      ...prev,
      metrics: [...prev.metrics, newMetric],
    }));
  };

  const updateMetric = (index: number, updates: Partial<MetricDefinition>) => {
    setFormData(prev => ({
      ...prev,
      metrics: prev.metrics.map((metric, i) => 
        i === index ? { ...metric, ...updates } : metric
      ),
    }));
  };

  const removeMetric = (index: number) => {
    setFormData(prev => ({
      ...prev,
      metrics: prev.metrics.filter((_, i) => i !== index),
    }));
  };

  // ============================================================================
  // RENDER FUNCTIONS
  // ============================================================================

  const renderStep1 = () => (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Experiment Name *
        </label>
        <input
          type="text"
          value={formData.name}
          onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          placeholder="e.g., New Checkout Flow Test"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Description
        </label>
        <textarea
          value={formData.description}
          onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
          rows={3}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          placeholder="Brief description of what you're testing..."
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Hypothesis *
        </label>
        <textarea
          value={formData.hypothesis}
          onChange={(e) => setFormData(prev => ({ ...prev, hypothesis: e.target.value }))}
          rows={3}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          placeholder="e.g., The new checkout flow will increase conversion rate by at least 5%..."
        />
        <p className="text-sm text-gray-500 mt-1">
          Clearly state what you expect to happen and why
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Confidence Level
          </label>
          <select
            value={formData.confidence_level}
            onChange={(e) => setFormData(prev => ({ ...prev, confidence_level: parseFloat(e.target.value) }))}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value={0.90}>90%</option>
            <option value={0.95}>95%</option>
            <option value={0.99}>99%</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Estimated Duration (days)
          </label>
          <input
            type="number"
            value={formData.estimated_duration_days || ''}
            onChange={(e) => setFormData(prev => ({ 
              ...prev, 
              estimated_duration_days: e.target.value ? parseInt(e.target.value) : undefined 
            }))}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder="14"
          />
        </div>
      </div>
    </div>
  );

  const renderStep2 = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium text-gray-900">Experiment Variants</h3>
        <button
          onClick={addVariant}
          className="flex items-center gap-2 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          <Plus className="w-4 h-4" />
          Add Variant
        </button>
      </div>

      <div className="space-y-4">
        {formData.variants.map((variant, index) => (
          <div key={variant.id} className="p-4 border rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={variant.is_control}
                  onChange={(e) => {
                    // Ensure only one control variant
                    if (e.target.checked) {
                      setFormData(prev => ({
                        ...prev,
                        variants: prev.variants.map((v, i) => ({
                          ...v,
                          is_control: i === index,
                        })),
                      }));
                    }
                  }}
                  className="rounded"
                />
                <label className="text-sm text-gray-600">Control Group</label>
              </div>
              
              {formData.variants.length > 2 && (
                <button
                  onClick={() => removeVariant(index)}
                  className="text-red-600 hover:text-red-800"
                >
                  <Minus className="w-4 h-4" />
                </button>
              )}
            </div>

            <div className="grid grid-cols-2 gap-4 mb-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Variant Name
                </label>
                <input
                  type="text"
                  value={variant.name}
                  onChange={(e) => updateVariant(index, { name: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Traffic %
                </label>
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={variant.traffic_percentage}
                  onChange={(e) => updateVariant(index, { traffic_percentage: parseFloat(e.target.value) || 0 })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Description
              </label>
              <textarea
                value={variant.description}
                onChange={(e) => updateVariant(index, { description: e.target.value })}
                rows={2}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Describe what's different in this variant..."
              />
            </div>
          </div>
        ))}
      </div>

      <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex items-center gap-2 text-blue-800">
          <Info className="w-4 h-4" />
          <span className="text-sm">
            Total traffic allocation: {formData.variants.reduce((sum, v) => sum + v.traffic_percentage, 0)}%
          </span>
        </div>
      </div>
    </div>
  );

  const renderStep3 = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium text-gray-900">Metrics to Track</h3>
        <button
          onClick={addMetric}
          className="flex items-center gap-2 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          <Plus className="w-4 h-4" />
          Add Metric
        </button>
      </div>

      <div className="space-y-4">
        {formData.metrics.map((metric, index) => (
          <div key={metric.id} className="p-4 border rounded-lg">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={metric.is_primary}
                  onChange={(e) => updateMetric(index, { is_primary: e.target.checked })}
                  className="rounded"
                />
                <label className="text-sm text-gray-600">Primary Metric</label>
              </div>
              
              {formData.metrics.length > 1 && (
                <button
                  onClick={() => removeMetric(index)}
                  className="text-red-600 hover:text-red-800"
                >
                  <Minus className="w-4 h-4" />
                </button>
              )}
            </div>

            <div className="grid grid-cols-2 gap-4 mb-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Metric Name
                </label>
                <input
                  type="text"
                  value={metric.name}
                  onChange={(e) => updateMetric(index, { name: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Metric Type
                </label>
                <select
                  value={metric.type}
                  onChange={(e) => updateMetric(index, { type: e.target.value as MetricDefinition['type'] })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="conversion_rate">Conversion Rate</option>
                  <option value="satisfaction_score">Satisfaction Score</option>
                  <option value="completion_rate">Completion Rate</option>
                  <option value="error_rate">Error Rate</option>
                  <option value="response_time">Response Time</option>
                  <option value="custom">Custom</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Description
              </label>
              <textarea
                value={metric.description}
                onChange={(e) => updateMetric(index, { description: e.target.value })}
                rows={2}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Describe how this metric is measured..."
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderStep4 = () => (
    <div className="space-y-6">
      <h3 className="text-lg font-medium text-gray-900">Sample Size Calculator</h3>
      
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Baseline Conversion Rate
          </label>
          <input
            type="number"
            min="0"
            max="1"
            step="0.01"
            value={sampleSizeCalc.baseline_conversion_rate}
            onChange={(e) => setSampleSizeCalc(prev => ({ 
              ...prev, 
              baseline_conversion_rate: parseFloat(e.target.value) || 0 
            }))}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <p className="text-xs text-gray-500 mt-1">Current conversion rate (0.0 - 1.0)</p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Minimum Effect Size
          </label>
          <input
            type="number"
            min="0"
            step="0.01"
            value={sampleSizeCalc.minimum_effect_size}
            onChange={(e) => setSampleSizeCalc(prev => ({ 
              ...prev, 
              minimum_effect_size: parseFloat(e.target.value) || 0 
            }))}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <p className="text-xs text-gray-500 mt-1">Minimum detectable difference</p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Statistical Power
          </label>
          <select
            value={sampleSizeCalc.statistical_power}
            onChange={(e) => setSampleSizeCalc(prev => ({ 
              ...prev, 
              statistical_power: parseFloat(e.target.value) 
            }))}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value={0.8}>80%</option>
            <option value={0.9}>90%</option>
            <option value={0.95}>95%</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Test Type
          </label>
          <select
            value={sampleSizeCalc.two_tailed ? 'two-tailed' : 'one-tailed'}
            onChange={(e) => setSampleSizeCalc(prev => ({ 
              ...prev, 
              two_tailed: e.target.value === 'two-tailed' 
            }))}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="two-tailed">Two-tailed</option>
            <option value="one-tailed">One-tailed</option>
          </select>
        </div>
      </div>

      <button
        onClick={calculateSampleSize}
        disabled={loading}
        className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
      >
        <Calculator className="w-4 h-4" />
        Calculate Sample Size
      </button>

      {calculatedSampleSize && (
        <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
          <h4 className="font-medium text-green-800 mb-2">Recommended Sample Size</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-green-700">Per Variant:</span>
              <span className="ml-2 font-bold">{calculatedSampleSize.sample_size_per_variant.toLocaleString()}</span>
            </div>
            <div>
              <span className="text-green-700">Total:</span>
              <span className="ml-2 font-bold">{calculatedSampleSize.total_sample_size.toLocaleString()}</span>
            </div>
          </div>
        </div>
      )}

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Target Sample Size (Optional)
        </label>
        <input
          type="number"
          value={formData.target_sample_size || ''}
          onChange={(e) => setFormData(prev => ({ 
            ...prev, 
            target_sample_size: e.target.value ? parseInt(e.target.value) : undefined 
          }))}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          placeholder="Use calculated value or enter custom"
        />
      </div>
    </div>
  );

  // ============================================================================
  // MAIN RENDER
  // ============================================================================

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-4xl max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Create New Experiment</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            ×
          </button>
        </div>

        {/* Step Indicator */}
        <div className="flex items-center mb-8">
          {[1, 2, 3, 4].map((step) => (
            <div key={step} className="flex items-center">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                step <= currentStep 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-200 text-gray-600'
              }`}>
                {step}
              </div>
              {step < 4 && (
                <div className={`w-12 h-1 mx-2 ${
                  step < currentStep ? 'bg-blue-600' : 'bg-gray-200'
                }`} />
              )}
            </div>
          ))}
        </div>

        {/* Step Content */}
        <div className="mb-8">
          {currentStep === 1 && renderStep1()}
          {currentStep === 2 && renderStep2()}
          {currentStep === 3 && renderStep3()}
          {currentStep === 4 && renderStep4()}
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center gap-2 text-red-800">
              <AlertCircle className="w-5 h-5" />
              <span>{error}</span>
            </div>
          </div>
        )}

        {/* Navigation */}
        <div className="flex justify-between">
          <button
            onClick={() => setCurrentStep(prev => Math.max(1, prev - 1))}
            disabled={currentStep === 1}
            className="px-4 py-2 text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
          >
            Previous
          </button>

          <div className="flex gap-2">
            {currentStep < 4 ? (
              <button
                onClick={() => setCurrentStep(prev => Math.min(4, prev + 1))}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Next
              </button>
            ) : (
              <button
                onClick={createExperiment}
                disabled={loading}
                className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
              >
                {loading ? (
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                ) : (
                  <Target className="w-4 h-4" />
                )}
                Create Experiment
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CreateExperimentForm; 