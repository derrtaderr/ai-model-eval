// Analytics data types matching backend API responses

export interface SystemOverview {
  time_range: string;
  start_time: string;
  end_time: string;
  evaluation_metrics: {
    total_evaluations: number;
    successful_evaluations: number;
    failed_evaluations: number;
    success_rate: number;
    average_score: number;
    throughput_per_hour: number;
    score_distribution: Record<string, number>;
  };
  batch_metrics: {
    total_jobs: number;
    active_jobs: number;
    completed_jobs: number;
    failed_jobs: number;
    average_completion_time_minutes: number;
    total_tasks_processed: number;
    system_utilization: number;
    success_rate: number;
    throughput_per_hour: number;
  };
  cost_metrics: {
    total_cost_usd: number;
    average_cost_per_evaluation: number;
    cost_per_evaluation: number;
    daily_cost_trend: Array<{ date: string; cost: number }>;
    cost_by_model: Record<string, number>;
    monthly_projection_usd: number;
  };
  performance_trends: {
    timestamps: string[];
    values: number[];
    labels: string[];
  };
  model_comparison: Array<{
    model_name: string;
    total_evaluations: number;
    success_rate: number;
    average_latency_ms: number;
    total_cost_usd: number;
    average_score: number;
    score_std_dev: number;
    cost_per_evaluation: number;
    throughput_per_hour: number;
    error_rate: number;
    calibration_accuracy?: number;
  }>;
  system_health: {
    status: string;
    uptime_seconds: number;
    memory_usage: string;
    cpu_usage: string;
    database_status: string;
    batch_processor_status: string;
    calibration_system_status: string;
  };
  alerts: Array<{
    id: string;
    level: string;
    title: string;
    message: string;
    metric_type: string;
    threshold_value: number;
    current_value: number;
    timestamp: string;
    resolved: boolean;
  }>;
  calibration_metrics: Record<string, unknown>;
}

export interface UserEngagement {
  time_period: string;
  start_time: string;
  end_time: string;
  user_metrics: {
    active_users: number;
    new_users: number;
    total_evaluations: number;
    avg_evaluations_per_user: number;
    user_growth_rate: number;
  };
  feature_usage: Array<{
    feature_name: string;
    usage_count: number;
    unique_users: number;
    adoption_rate: number;
    avg_usage_per_user: number;
  }>;
  user_journeys: Array<{
    user_email: string;
    first_evaluation_date: string;
    total_evaluations: number;
    days_since_first_evaluation: number;
    onboarding_completed: boolean;
  }>;
  engagement_trends: {
    daily_evaluations: Array<{ date: string; count: number }>;
    daily_active_users: Array<{ date: string; count: number }>;
  };
  retention_metrics: {
    day_1_retention: number;
    day_7_retention: number;
    day_30_retention: number;
    cohort_analysis: string;
  };
}

export interface AgreementAnalysis {
  time_period: string;
  total_comparisons: number;
  agreement_rate: number;
  strong_agreement_rate: number;
  disagreement_patterns: {
    ai_higher: number;
    human_higher: number;
    avg_disagreement: number;
  };
  confidence_correlation: number;
  bias_indicators: Record<string, number>;
  model_reliability_scores: Record<string, number>;
}

export interface AcceptanceRates {
  time_period: string;
  total_ai_suggestions: number;
  accepted_suggestions: number;
  rejected_suggestions: number;
  acceptance_rate: number;
  acceptance_by_confidence: {
    high: number;
    medium: number;
    low: number;
  };
  acceptance_by_criteria: Record<string, number>;
  trust_trend_over_time: Array<{
    timestamp: string;
    trust_score: number;
  }>;
}

export interface ModelComparison {
  model_name: string;
  total_evaluations: number;
  success_rate: number;
  average_latency_ms: number;
  total_cost_usd: number;
  average_score: number;
  score_std_dev: number;
  cost_per_evaluation: number;
  throughput_per_hour: number;
  error_rate: number;
  calibration_accuracy?: number;
}

export interface SystemHealth {
  status: string;
  uptime_seconds: number;
  memory_usage: string;
  cpu_usage: string;
  database_status: string;
  batch_processor_status: string;
  calibration_system_status: string;
}

export interface Alert {
  id: string;
  level: 'info' | 'success' | 'warning' | 'error' | 'critical';
  title: string;
  message: string;
  metric_type: string;
  threshold_value: number;
  current_value: number;
  timestamp: string;
  resolved: boolean;
}

export interface RealTimeUpdate {
  type: 'metric_update' | 'alert_triggered' | 'batch_progress' | 'user_activity' | 'system_status' | 'cost_update';
  data: Record<string, unknown>;
  level: 'info' | 'success' | 'warning' | 'error' | 'critical';
  title?: string;
  message?: string;
  timestamp: string;
} 