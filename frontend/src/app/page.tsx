'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Upload,
  Download,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CheckCircle,
  XCircle,
  Clock,
  User,
  Bot,
  AlertCircle,
  FileText,
  X,
  BarChart3
} from 'lucide-react';

interface Trace {
  id: string;
  timestamp: string;
  tool: string;
  scenario: string;
  status: 'pending' | 'accepted' | 'rejected';
  modelScore: 'pass' | 'fail';
  humanScore: 'good' | 'bad' | null;
  dataSource: 'human' | 'synthetic';
  conversation: {
    userInput: string;
    aiResponse: string;
    systemPrompt?: string;
  };
  functions: {
    name: string;
    parameters: Record<string, unknown>;
    result: Record<string, unknown>;
    executionTime: number;
  }[];
  metadata: {
    modelName: string;
    latencyMs: number;
    tokenCount: { input: number; output: number };
    costUsd: number;
    temperature: number;
    maxTokens: number;
  };
}

// Dynamic filter options discovered from actual data
interface FilterOptions {
  tools: string[];
  scenarios: string[];
  statuses: string[];
  dataSources: string[];
}

export default function EvaluationDashboard() {
  const [traces, setTraces] = useState<Trace[]>([]);
  const [filters, setFilters] = useState({
    tool: 'All Tools',
    scenario: 'All Scenarios', 
    status: 'All Status',
    dataSource: 'All Sources'
  });
  const [filterOptions, setFilterOptions] = useState<FilterOptions>({
    tools: ['All Tools'],
    scenarios: ['All Scenarios'],
    statuses: ['All Status', 'pending', 'accepted', 'rejected'],
    dataSources: ['All Sources', 'human', 'synthetic']
  });
  const [selectedTrace, setSelectedTrace] = useState<Trace | null>(null);
  const [activeTab, setActiveTab] = useState<'chat' | 'functions' | 'metadata'>('chat');
  const [evaluationNotes, setEvaluationNotes] = useState('');
  const [showRejectionModal, setShowRejectionModal] = useState(false);
  const [evaluationFeedback, setEvaluationFeedback] = useState<string | null>(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Mock API functions (replace with real API calls when backend is ready)
  const mockApiCall = async (endpoint: string): Promise<Trace[] | { success: boolean; message: string; tracesProcessed?: number; newTraces?: number }> => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    if (endpoint === '/api/evaluations/traces') {
      // Return empty array initially - data comes from uploads
      return [];
    }
    
    if (endpoint.includes('/api/evaluations/submit')) {
      // Simulate successful evaluation submission
      return { success: true, message: 'Evaluation submitted successfully' };
    }
    
    if (endpoint.includes('/api/evaluations/upload')) {
      // Simulate successful upload
      return { 
        success: true, 
        message: 'Data uploaded successfully',
        tracesProcessed: 5,
        newTraces: 3
      };
    }
    
    return { success: true, message: 'Operation completed' };
  };

  const loadTraces = useCallback(async () => {
    try {
      const data = await mockApiCall('/api/evaluations');
      if (Array.isArray(data)) {
        setTraces(data);
        
        // Update filter options based on loaded data
        const tools = Array.from(new Set(data.map(trace => trace.tool)));
        const scenarios = Array.from(new Set(data.map(trace => trace.scenario)));
        
        setFilterOptions({
          tools: ['All Tools', ...tools],
          scenarios: ['All Scenarios', ...scenarios],
          statuses: ['All Status', 'pending', 'accepted', 'rejected'],
          dataSources: ['All Sources', 'human', 'synthetic']
        });
      }
    } catch (error) {
      console.error('Failed to load traces:', error);
    }
  }, []);

  // Load data on component mount
  useEffect(() => {
    loadTraces();
  }, [loadTraces]);

  // Filter traces based on selected filters
  const filteredTraces = traces.filter(trace => {
    if (filters.tool !== 'All Tools' && trace.tool !== filters.tool) return false;
    if (filters.scenario !== 'All Scenarios' && trace.scenario !== filters.scenario) return false;
    if (filters.status !== 'All Status' && trace.status !== filters.status) return false;
    if (filters.dataSource !== 'All Sources' && trace.dataSource !== filters.dataSource) return false;
    return true;
  });

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Reset previous errors
    setUploadError(null);

    // Validate file type
    const allowedTypes = ['application/json', 'text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'];
    const allowedExtensions = ['.json', '.csv', '.xlsx', '.xls'];
    const fileExtension = file.name.toLowerCase().slice(file.name.lastIndexOf('.'));
    
    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
      setUploadError('Please upload a valid file (JSON, CSV, or Excel)');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setUploadError('File size must be less than 10MB');
      return;
    }

    setUploadFile(file);
    setUploadError(null);
    console.log('File selected:', file.name, file.type, file.size);
  };

  // Process uploaded file
  const processUpload = async () => {
    if (!uploadFile) {
      setUploadError('Please select a file first');
      return;
    }

    let progressInterval: NodeJS.Timeout | undefined;
    
    try {
      setIsUploading(true);
      setUploadError(null);
      setUploadProgress(0);
      
      console.log('Starting upload process for:', uploadFile.name);
      
      // Simulate progress
      progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90));
      }, 100);

      const text = await uploadFile.text();
      console.log('File content loaded, length:', text.length);
      
      let parsedData: Trace[] = [];

      if (uploadFile.name.toLowerCase().endsWith('.json')) {
        try {
          const jsonData = JSON.parse(text);
          console.log('JSON parsed successfully:', jsonData);
          
          // Handle both single trace and array of traces
          const rawTraces = Array.isArray(jsonData) ? jsonData : [jsonData];
          console.log('Processing traces:', rawTraces.length);
          
          // Validate and ensure each trace has required fields
          parsedData = rawTraces.map((trace: Record<string, unknown>, index) => {
            console.log(`Processing trace ${index}:`, trace);
            
            return {
              id: trace.id as string || `uploaded-${Date.now()}-${index}`,
              timestamp: trace.timestamp as string || new Date().toLocaleString(),
              tool: trace.tool as string || 'Unknown Tool',
              scenario: trace.scenario as string || 'Unknown Scenario',
              status: (trace.status as 'pending' | 'accepted' | 'rejected') || 'pending',
              modelScore: (trace.modelScore as 'pass' | 'fail') || 'pass',
              humanScore: (trace.humanScore as 'good' | 'bad' | null) || null,
              dataSource: (trace.dataSource as 'human' | 'synthetic') || 'human',
              conversation: {
                userInput: (trace.conversation as Record<string, unknown>)?.userInput as string || trace.userInput as string || 'No input provided',
                aiResponse: (trace.conversation as Record<string, unknown>)?.aiResponse as string || trace.aiResponse as string || 'No response provided',
                systemPrompt: (trace.conversation as Record<string, unknown>)?.systemPrompt as string || trace.systemPrompt as string || ''
              },
              functions: ((trace.functions as Record<string, unknown>[]) || []).map((func: Record<string, unknown>) => ({
                name: func.name as string || 'unknown',
                parameters: func.parameters as Record<string, unknown> || {},
                result: func.result as Record<string, unknown> || {},
                executionTime: func.executionTime as number || 0
              })),
              metadata: {
                modelName: ((trace.metadata as Record<string, unknown>)?.modelName as string) || 'Unknown Model',
                latencyMs: ((trace.metadata as Record<string, unknown>)?.latencyMs as number) || 0,
                tokenCount: {
                  input: (((trace.metadata as Record<string, unknown>)?.tokenCount as Record<string, unknown>)?.input as number) || 0,
                  output: (((trace.metadata as Record<string, unknown>)?.tokenCount as Record<string, unknown>)?.output as number) || 0
                },
                costUsd: ((trace.metadata as Record<string, unknown>)?.costUsd as number) || 0,
                temperature: ((trace.metadata as Record<string, unknown>)?.temperature as number) || 0.7,
                maxTokens: ((trace.metadata as Record<string, unknown>)?.maxTokens as number) || 1000
              }
            };
          });
          
        } catch (jsonError) {
          console.error('JSON parsing error:', jsonError);
          throw new Error('Invalid JSON format. Please check your file structure.');
        }
      } else if (uploadFile.name.toLowerCase().endsWith('.csv')) {
        // Basic CSV parsing - in production you'd use a proper CSV parser
        console.log('CSV file detected, but parsing not implemented yet');
        setUploadError('CSV parsing is coming soon. Please use JSON format for now.');
        return;
      } else {
        setUploadError('Unsupported file format. Please use JSON or CSV.');
        return;
      }

      console.log('Parsed data:', parsedData);

      // Simulate API call
      await mockApiCall('/api/evaluations/upload');
      
      // Add parsed data to existing traces
      setTraces(prevTraces => {
        const newTraces = [...prevTraces, ...parsedData];
        console.log('Updated traces:', newTraces.length);
        return newTraces;
      });
      
      // Update filter options with new data
      const allTraces = [...traces, ...parsedData];
      const tools = Array.from(new Set(allTraces.map(t => t.tool)));
      const scenarios = Array.from(new Set(allTraces.map(t => t.scenario)));
      
      setFilterOptions({
        tools: ['All Tools', ...tools],
        scenarios: ['All Scenarios', ...scenarios],
        statuses: ['All Status', 'pending', 'accepted', 'rejected'],
        dataSources: ['All Sources', 'human', 'synthetic']
      });

      if (progressInterval) clearInterval(progressInterval);
      setUploadProgress(100);
      
      console.log(`Upload successful! Added ${parsedData.length} traces.`);
      
      // Close modal after success
      setTimeout(() => {
        setShowUploadModal(false);
        setUploadFile(null);
        setUploadProgress(0);
        setIsUploading(false);
      }, 1000);

    } catch (error) {
      console.error('Upload error:', error);
      setUploadError(error instanceof Error ? error.message : 'Upload failed. Please try again.');
      setIsUploading(false);
      setUploadProgress(0);
      if (progressInterval) clearInterval(progressInterval);
    }
  };

  const downloadLabeledData = async () => {
    setIsDownloading(true);
    
    try {
      // Check if we have any traces to export
      if (traces.length === 0) {
        alert('No trace data available to download. Please upload some data first.');
        setIsDownloading(false);
        return;
      }
      
      // Generate CSV header
      const csvHeader = `trace_id,timestamp,tool,scenario,status,model_score,human_score,data_source,user_input,ai_response,model_name,latency_ms,token_count_input,token_count_output,cost_usd,temperature\n`;
      
      // Convert traces to CSV format
      const csvRows = traces.map(trace => {
        // Escape CSV fields that might contain commas or quotes
        const escapeCSV = (field: string | null | undefined) => {
          if (field === null || field === undefined) return '';
          const stringField = String(field);
          if (stringField.includes(',') || stringField.includes('"') || stringField.includes('\n')) {
            return `"${stringField.replace(/"/g, '""')}"`;
          }
          return stringField;
        };
        
        return [
          escapeCSV(trace.id),
          escapeCSV(trace.timestamp),
          escapeCSV(trace.tool),
          escapeCSV(trace.scenario),
          escapeCSV(trace.status),
          escapeCSV(trace.modelScore),
          escapeCSV(trace.humanScore),
          escapeCSV(trace.dataSource),
          escapeCSV(trace.conversation.userInput),
          escapeCSV(trace.conversation.aiResponse),
          escapeCSV(trace.metadata.modelName),
          escapeCSV(trace.metadata.latencyMs.toString()),
          escapeCSV(trace.metadata.tokenCount.input.toString()),
          escapeCSV(trace.metadata.tokenCount.output.toString()),
          escapeCSV(trace.metadata.costUsd.toString()),
          escapeCSV(trace.metadata.temperature.toString())
        ].join(',');
      }).join('\n');
      
      // Combine header and data
      const csvContent = csvHeader + csvRows;
      
      // Create and download the file
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `llm-evaluation-data-${new Date().toISOString().split('T')[0]}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      console.log(`Downloaded ${traces.length} traces as CSV`);
    } catch (error) {
      console.error('Download failed:', error);
      alert('Download failed. Please try again.');
    } finally {
      setIsDownloading(false);
    }
  };

  const SimpleChart = ({ data, title, color }: { data: typeof agreementData, title: string, color: string }) => {
    if (data.length === 0) {
      return (
        <div className="bg-white p-6 rounded-lg border">
          <h3 className="text-sm font-medium text-gray-600 mb-4 flex items-center gap-2">
            {title}
            <div className="w-2 h-2 rounded-full bg-gray-400"></div>
          </h3>
          <div className="h-32 flex items-center justify-center">
            <div className="text-center">
              <div className="text-gray-700 text-sm">No data available</div>
              <div className="text-gray-600 text-xs mt-1">Start evaluating traces to see trends</div>
            </div>
          </div>
        </div>
      );
    }

    const maxRate = Math.max(...data.map(d => d.rate));
    const minRate = Math.min(...data.map(d => d.rate));
    const range = maxRate - minRate || 1;

    // Handle single data point case
    const getXPosition = (index: number) => {
      if (data.length === 1) return 150; // Center the single point
      return (index / (data.length - 1)) * 280 + 10;
    };

    const getYPosition = (rate: number) => {
      return 120 - ((rate - minRate) / range) * 100;
    };

    return (
      <div className="bg-white p-6 rounded-lg border">
        <h3 className="text-sm font-medium text-gray-600 mb-4 flex items-center gap-2">
          {title}
          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }}></div>
        </h3>
        <div className="h-32 relative">
          <div className="absolute left-0 top-0 text-xs text-gray-700">100%</div>
          <div className="absolute left-0 top-1/2 text-xs text-gray-700">75%</div>
          <div className="absolute left-0 bottom-8 text-xs text-gray-700">50%</div>
          <div className="absolute left-0 bottom-0 text-xs text-gray-700">0%</div>
          
          <svg className="w-full h-full ml-6" viewBox="0 0 300 120">
            {/* Only draw line if we have more than one point */}
            {data.length > 1 && (
              <polyline
                fill="none"
                stroke={color}
                strokeWidth="2"
                points={data.map((point, index) => 
                  `${getXPosition(index)},${getYPosition(point.rate)}`
                ).join(' ')}
              />
            )}
            
            {/* Draw circles for each data point */}
            {data.map((point, index) => (
              <circle
                key={index}
                cx={getXPosition(index)}
                cy={getYPosition(point.rate)}
                r="3"
                fill={color}
              />
            ))}
          </svg>
          
          <div className="flex justify-between mt-2 ml-6 text-xs text-gray-700">
            {data.length === 1 ? (
              <div className="w-full text-center">{data[0].date}</div>
            ) : (
              data.map((point, index) => (
                <span key={index}>{point.date}</span>
              ))
            )}
          </div>
        </div>
      </div>
    );
  };

  // Simple dropdown component for filters
  const FilterDropdown = ({ value, options, onChange }: { value: string, options: string[], onChange: (value: string) => void }) => {
    const [isOpen, setIsOpen] = useState(false);
    
    return (
      <div className="relative">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center gap-2 bg-gray-50 border border-gray-200 rounded-lg px-4 py-2 text-gray-700 hover:bg-gray-100 min-w-[140px] justify-between"
        >
          <span className="text-sm">{value}</span>
          <ChevronDown className="w-4 h-4" />
        </button>
        
        {isOpen && (
          <div className="absolute top-full left-0 mt-1 w-full bg-white border rounded-lg shadow-lg z-10 max-h-60 overflow-y-auto">
            {options.map((option) => (
              <button
                key={option}
                onClick={() => {
                  onChange(option);
                  setIsOpen(false);
                }}
                className="w-full text-left px-3 py-2 hover:bg-gray-50 text-sm text-gray-900 first:rounded-t-lg last:rounded-b-lg"
              >
                {option === value && <span className="mr-2">✓</span>}
                {option}
              </button>
            ))}
          </div>
        )}
      </div>
    );
  };

  // Handle evaluation actions
  const handleEvaluation = (action: 'accept' | 'reject' | 'review', notes?: string) => {
    if (!selectedTrace) return;

    // For reject action, show modal first
    if (action === 'reject') {
      setShowRejectionModal(true);
      return;
    }

    // Update the trace in the traces array
    setTraces(prevTraces => 
      prevTraces.map(trace => 
        trace.id === selectedTrace.id 
          ? { 
              ...trace, 
              status: action === 'accept' ? 'accepted' : 'pending',
              humanScore: action === 'accept' ? 'good' : null
            }
          : trace
      )
    );

    // Update selected trace
    setSelectedTrace(prev => prev ? {
      ...prev,
      status: action === 'accept' ? 'accepted' : 'pending',
      humanScore: action === 'accept' ? 'good' : null
    } : null);

    // Show feedback
    const actionText = action === 'accept' ? 'accepted' : 'marked for review';
    setEvaluationFeedback(`Trace ${selectedTrace.id} has been ${actionText}`);
    
    // Clear feedback after 3 seconds
    setTimeout(() => setEvaluationFeedback(null), 3000);

    // Clear notes after evaluation
    setEvaluationNotes('');

    console.log(`Evaluated trace ${selectedTrace.id} as ${action}`, notes ? `with notes: ${notes}` : '');
  };

  const handleRejectionSubmit = (reason: string) => {
    if (!selectedTrace) return;

    // Update the trace with rejection
    setTraces(prevTraces => 
      prevTraces.map(trace => 
        trace.id === selectedTrace.id 
          ? { 
              ...trace, 
              status: 'rejected',
              humanScore: 'bad'
            }
          : trace
      )
    );

    // Update selected trace
    setSelectedTrace(prev => prev ? {
      ...prev,
      status: 'rejected',
      humanScore: 'bad'
    } : null);

    // Show feedback
    setEvaluationFeedback(`Trace ${selectedTrace.id} has been rejected: ${reason}`);
    
    // Clear feedback after 3 seconds
    setTimeout(() => setEvaluationFeedback(null), 3000);

    setShowRejectionModal(false);
    console.log(`Rejected trace ${selectedTrace.id} with reason: ${reason}`);
  };

  // Calculate real chart data from traces
  const calculateChartData = useCallback(() => {
    if (traces.length === 0) return { agreementData: [], acceptanceData: [] };

    // Calculate agreement rate (how often human and model scores agree)
    const tracesWithHumanScores = traces.filter(t => t.humanScore !== null);
    const agreementRate = tracesWithHumanScores.length > 0 
      ? (tracesWithHumanScores.filter(t => 
          (t.modelScore === 'pass' && t.humanScore === 'good') ||
          (t.modelScore === 'fail' && t.humanScore === 'bad')
        ).length / tracesWithHumanScores.length) * 100
      : 0;

    // Calculate human acceptance rate
    const acceptanceRate = tracesWithHumanScores.length > 0
      ? (tracesWithHumanScores.filter(t => t.humanScore === 'good').length / tracesWithHumanScores.length) * 100
      : 0;

    // Generate simple data points (in a real app, this would be time-series data)
    const today = new Date().toLocaleDateString();
    
    return {
      agreementData: [{ date: today, rate: agreementRate }],
      acceptanceData: [{ date: today, rate: acceptanceRate }]
    };
  }, [traces]);

  const { agreementData, acceptanceData } = calculateChartData();

  // Navigation functions for trace details
  const getCurrentTraceIndex = () => {
    if (!selectedTrace) return -1;
    return filteredTraces.findIndex(trace => trace.id === selectedTrace.id);
  };

  const navigateToTrace = (direction: 'prev' | 'next') => {
    const currentIndex = getCurrentTraceIndex();
    if (currentIndex === -1) return;

    let newIndex;
    if (direction === 'prev') {
      newIndex = currentIndex > 0 ? currentIndex - 1 : filteredTraces.length - 1;
    } else {
      newIndex = currentIndex < filteredTraces.length - 1 ? currentIndex + 1 : 0;
    }

    setSelectedTrace(filteredTraces[newIndex]);
    setActiveTab('chat');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-20">
            <div className="flex-1">
              <h1 className="text-2xl font-bold text-gray-900">LLM Evaluation Dashboard</h1>
              <p className="text-sm text-gray-600 mt-1">Trace management and human evaluation interface. Visit Analytics Dashboard for detailed metrics and insights.</p>
            </div>
            <div className="flex items-center gap-4 ml-8">
              {/* Analytics Dashboard Link */}
              <a 
                href="/analytics"
                className="bg-purple-600 text-white px-6 py-3 rounded-lg flex items-center gap-2 hover:bg-purple-700 transition-colors font-medium"
                title="Advanced analytics with detailed metrics, trends, and system health monitoring"
              >
                <BarChart3 className="w-5 h-5" />
                Analytics Dashboard
              </a>

              {/* Upload Data Button */}
              <button 
                className="bg-blue-600 text-white px-6 py-3 rounded-lg flex items-center gap-2 hover:bg-blue-700 transition-colors font-medium"
                onClick={() => setShowUploadModal(true)}
                title="Upload trace data in JSON or CSV format"
              >
                <Upload className="w-5 h-5" />
                Upload Data
              </button>

              {/* Download Labeled Data Button */}
              <button 
                className="bg-green-600 text-white px-6 py-3 rounded-lg flex items-center gap-2 hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium"
                onClick={downloadLabeledData}
                disabled={isDownloading}
                title="Download evaluation results as CSV"
              >
                <Download className="w-5 h-5" />
                {isDownloading ? 'Downloading...' : 'Download Labeled Data'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Filter Section */}
        <div className="bg-white rounded-lg border border-gray-200 p-4 mb-6">
          <h3 className="text-sm font-medium text-gray-900 mb-4">Filter Records</h3>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Tool Filter */}
            <div>
              <label className="block text-xs text-gray-600 mb-1">Tool</label>
              <FilterDropdown
                value={filters.tool}
                options={filterOptions.tools}
                onChange={(value) => setFilters({...filters, tool: value})}
              />
            </div>

            {/* Scenario Filter */}
            <div>
              <label className="block text-xs text-gray-600 mb-1">Scenario</label>
              <FilterDropdown
                value={filters.scenario}
                options={filterOptions.scenarios}
                onChange={(value) => setFilters({...filters, scenario: value})}
              />
            </div>

            {/* Status Filter */}
            <div>
              <label className="block text-xs text-gray-600 mb-1">Status</label>
              <FilterDropdown
                value={filters.status}
                options={filterOptions.statuses}
                onChange={(value) => setFilters({...filters, status: value})}
              />
            </div>

            {/* Data Source Filter */}
            <div>
              <label className="block text-xs text-gray-600 mb-1">Data Source</label>
              <FilterDropdown
                value={filters.dataSource}
                options={filterOptions.dataSources}
                onChange={(value) => setFilters({...filters, dataSource: value})}
              />
            </div>
          </div>
        </div>

        {/* Analytics Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <SimpleChart 
            data={agreementData} 
            title="LLM <-> Human Agreement Rate" 
            color="#8b5cf6" 
          />
          <SimpleChart 
            data={acceptanceData} 
            title="Human Acceptance Rate" 
            color="#3b82f6" 
          />
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Left Column - Trace List */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-gray-900">Trace Records</h3>
            
            {filteredTraces.length > 0 ? (
              <div className="space-y-3">
                {filteredTraces.map((trace) => (
                  <div
                    key={trace.id}
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                      selectedTrace?.id === trace.id 
                        ? 'border-blue-500 bg-blue-50' 
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => {
                      setSelectedTrace(trace);
                      setActiveTab('chat');
                    }}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3">
                        <div className="text-sm font-medium text-gray-900">
                          {trace.id}
                        </div>
                        <div className="text-xs text-gray-500">
                          {trace.timestamp}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded">
                          {trace.tool}
                        </span>
                        <span className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded">
                          {trace.scenario}
                        </span>
                      </div>
                    </div>
                    <div className="text-sm text-gray-600 line-clamp-2">
                      {trace.conversation.userInput}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-8 text-center bg-white rounded-lg border">
                <div className="text-gray-500 mb-2">No traces available</div>
                <div className="text-sm text-gray-400">Use the &quot;Upload Data&quot; button above to get started</div>
              </div>
            )}
          </div>

          {/* Right Column - Trace Detail */}
          <div className="bg-white rounded-lg border">
            {/* Records Navigation */}
            <div className="p-4 border-b">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">
                  Record {getCurrentTraceIndex() + 1} of {filteredTraces.length}
                </span>
                <div className="flex items-center gap-2">
                  <div className="flex items-center gap-2">
                    <button 
                      className="p-1 border rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                      onClick={() => navigateToTrace('prev')}
                      disabled={filteredTraces.length <= 1}
                    >
                      <ChevronLeft className="w-4 h-4" />
                    </button>
                    <span className="text-sm text-gray-600">Previous</span>
                    <span className="text-sm text-gray-600">Next</span>
                    <button 
                      className="p-1 border rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                      onClick={() => navigateToTrace('next')}
                      disabled={filteredTraces.length <= 1}
                    >
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Tab Navigation */}
            <div className="border-b">
              <div className="flex">
                {(['chat', 'functions', 'metadata'] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`px-6 py-3 text-sm font-medium border-b-2 capitalize ${
                      activeTab === tab 
                        ? 'border-blue-500 text-blue-600' 
                        : 'border-transparent text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    {tab}
                  </button>
                ))}
              </div>
            </div>

            {/* Tab Content */}
            <div className="p-6">
              {selectedTrace ? (
                <div className="space-y-6">
                  {/* Trace Overview - Always visible */}
                  <div className="grid grid-cols-2 gap-4 text-sm border-b pb-4">
                    <div>
                      <span className="text-gray-600">Trace ID:</span>
                      <span className="ml-2 font-mono text-gray-900">{selectedTrace.id}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Timestamp:</span>
                      <span className="ml-2 text-gray-900">{selectedTrace.timestamp}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Tool:</span>
                      <span className="ml-2 text-gray-900">{selectedTrace.tool}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Scenario:</span>
                      <span className="ml-2 text-gray-900">{selectedTrace.scenario}</span>
                    </div>
                  </div>

                  {/* Tab-specific Content */}
                  {activeTab === 'chat' && (
                    <div className="space-y-4">
                      {/* System Prompt */}
                      {selectedTrace.conversation.systemPrompt && (
                        <div className="bg-yellow-50 border border-yellow-200 p-4 rounded-lg">
                          <div className="flex items-center gap-2 text-sm text-yellow-800 font-medium mb-2">
                            <AlertCircle className="w-4 h-4" />
                            System Prompt
                          </div>
                          <div className="text-sm text-yellow-700 whitespace-pre-wrap">
                            {selectedTrace.conversation.systemPrompt}
                          </div>
                        </div>
                      )}

                      {/* Conversation */}
                      <div className="space-y-4">
                        <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg">
                          <div className="flex items-center gap-2 text-sm text-blue-800 font-medium mb-2">
                            <User className="w-4 h-4" />
                            User Input
                          </div>
                          <div className="text-sm text-blue-700 whitespace-pre-wrap">
                            {selectedTrace.conversation.userInput}
                          </div>
                        </div>
                        
                        <div className="bg-gray-50 border border-gray-200 p-4 rounded-lg">
                          <div className="flex items-center gap-2 text-sm text-gray-800 font-medium mb-2">
                            <Bot className="w-4 h-4" />
                            AI Response
                          </div>
                          <div className="text-sm text-gray-700 whitespace-pre-wrap">
                            {selectedTrace.conversation.aiResponse}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {activeTab === 'functions' && (
                    <div className="space-y-4">
                      {selectedTrace.functions && selectedTrace.functions.length > 0 ? (
                        selectedTrace.functions.map((func, index) => (
                          <div key={index} className="bg-purple-50 border border-purple-200 p-4 rounded-lg">
                            <div className="flex items-center justify-between mb-3">
                              <h4 className="text-sm font-medium text-purple-900">{func.name}</h4>
                              <span className="text-xs text-purple-600 bg-purple-100 px-2 py-1 rounded">
                                {func.executionTime}ms
                              </span>
                            </div>
                            
                            <div className="space-y-3 text-sm">
                              <div>
                                <span className="font-medium text-purple-800">Parameters:</span>
                                <pre className="mt-1 bg-purple-100 p-2 rounded text-xs overflow-x-auto">
                                  {JSON.stringify(func.parameters, null, 2)}
                                </pre>
                              </div>
                              
                              <div>
                                <span className="font-medium text-purple-800">Result:</span>
                                <pre className="mt-1 bg-purple-100 p-2 rounded text-xs overflow-x-auto">
                                  {JSON.stringify(func.result, null, 2)}
                                </pre>
                              </div>
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="text-center py-8 text-gray-500">
                          <div className="text-sm">No function calls in this trace</div>
                        </div>
                      )}
                    </div>
                  )}

                  {activeTab === 'metadata' && (
                    <div className="space-y-4">
                      {/* Model Information */}
                      <div className="bg-indigo-50 border border-indigo-200 p-4 rounded-lg">
                        <h4 className="text-sm font-medium text-indigo-900 mb-3">Model Information</h4>
                        <div className="grid grid-cols-2 gap-3 text-sm">
                          <div>
                            <span className="text-indigo-700 font-medium">Model:</span>
                            <span className="ml-2 text-indigo-800">{selectedTrace.metadata.modelName}</span>
                          </div>
                          <div>
                            <span className="text-indigo-700 font-medium">Temperature:</span>
                            <span className="ml-2 text-indigo-800">{selectedTrace.metadata.temperature}</span>
                          </div>
                          <div>
                            <span className="text-indigo-700 font-medium">Max Tokens:</span>
                            <span className="ml-2 text-indigo-800">{selectedTrace.metadata.maxTokens}</span>
                          </div>
                        </div>
                      </div>

                      {/* Performance Metrics */}
                      <div className="bg-green-50 border border-green-200 p-4 rounded-lg">
                        <h4 className="text-sm font-medium text-green-900 mb-3">Performance Metrics</h4>
                        <div className="grid grid-cols-2 gap-3 text-sm">
                          <div>
                            <span className="text-green-700 font-medium">Latency:</span>
                            <span className="ml-2 text-green-800">{selectedTrace.metadata.latencyMs}ms</span>
                          </div>
                          <div>
                            <span className="text-green-700 font-medium">Cost:</span>
                            <span className="ml-2 text-green-800">${selectedTrace.metadata.costUsd.toFixed(4)}</span>
                          </div>
                        </div>
                      </div>

                      {/* Token Usage */}
                      <div className="bg-orange-50 border border-orange-200 p-4 rounded-lg">
                        <h4 className="text-sm font-medium text-orange-900 mb-3">Token Usage</h4>
                        <div className="grid grid-cols-3 gap-3 text-sm">
                          <div>
                            <span className="text-orange-700 font-medium">Input:</span>
                            <span className="ml-2 text-orange-800">{selectedTrace.metadata.tokenCount.input}</span>
                          </div>
                          <div>
                            <span className="text-orange-700 font-medium">Output:</span>
                            <span className="ml-2 text-orange-800">{selectedTrace.metadata.tokenCount.output}</span>
                          </div>
                          <div>
                            <span className="text-orange-700 font-medium">Total:</span>
                            <span className="ml-2 text-orange-800 font-semibold">
                              {selectedTrace.metadata.tokenCount.input + selectedTrace.metadata.tokenCount.output}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Status Information */}
                      <div className="bg-gray-50 border border-gray-200 p-4 rounded-lg">
                        <h4 className="text-sm font-medium text-gray-900 mb-3">Evaluation Status</h4>
                        <div className="grid grid-cols-2 gap-3 text-sm">
                          <div>
                            <span className="text-gray-700 font-medium">Model Score:</span>
                            <span className={`ml-2 px-2 py-1 rounded text-xs ${
                              selectedTrace.modelScore === 'pass' 
                                ? 'bg-green-100 text-green-800' 
                                : 'bg-red-100 text-red-800'
                            }`}>
                              {selectedTrace.modelScore}
                            </span>
                          </div>
                          <div>
                            <span className="text-gray-700 font-medium">Human Score:</span>
                            <span className={`ml-2 px-2 py-1 rounded text-xs ${
                              selectedTrace.humanScore === 'good' ? 'bg-green-100 text-green-800' :
                              selectedTrace.humanScore === 'bad' ? 'bg-red-100 text-red-800' :
                              'bg-gray-100 text-gray-600'
                            }`}>
                              {selectedTrace.humanScore || 'pending'}
                            </span>
                          </div>
                          <div>
                            <span className="text-gray-700 font-medium">Status:</span>
                            <span className={`ml-2 px-2 py-1 rounded text-xs ${
                              selectedTrace.status === 'accepted' ? 'bg-green-100 text-green-800' :
                              selectedTrace.status === 'rejected' ? 'bg-red-100 text-red-800' :
                              'bg-yellow-100 text-yellow-800'
                            }`}>
                              {selectedTrace.status}
                            </span>
                          </div>
                          <div>
                            <span className="text-gray-700 font-medium">Data Source:</span>
                            <span className="ml-2 text-gray-800">{selectedTrace.dataSource}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Evaluation Controls - Always visible at bottom */}
                  <div className="bg-gray-50 border border-gray-200 p-4 rounded-lg mt-6">
                    {/* Evaluation Feedback */}
                    {evaluationFeedback && (
                      <div className="bg-green-50 border border-green-200 text-green-700 p-3 rounded-lg mb-4 text-sm">
                        ✓ {evaluationFeedback}
                      </div>
                    )}
                    
                    <h4 className="text-sm font-medium text-gray-900 mb-3">Human Evaluation</h4>
                    <div className="flex gap-3">
                      <button 
                        className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                        onClick={() => handleEvaluation('accept')}
                      >
                        <CheckCircle className="w-4 h-4" />
                        Accept
                      </button>
                      <button 
                        className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                        onClick={() => handleEvaluation('reject')}
                      >
                        <XCircle className="w-4 h-4" />
                        Reject
                      </button>
                      <button 
                        className="flex items-center gap-2 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                        onClick={() => handleEvaluation('review')}
                      >
                        <Clock className="w-4 h-4" />
                        Mark for Review
                      </button>
                    </div>
                    <div className="mt-3">
                      <label className="block text-xs text-gray-600 mb-1">Evaluation Notes (Optional)</label>
                      <textarea 
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm text-gray-900 placeholder-gray-500"
                        rows={2}
                        placeholder="Add notes about your evaluation..."
                        value={evaluationNotes}
                        onChange={(e) => setEvaluationNotes(e.target.value)}
                      />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <div className="text-gray-500 mb-4">Select a trace record to view details</div>
                  <div className="text-sm text-gray-400">Choose a trace from the left panel to see the conversation, function calls, and metadata</div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Upload Trace Data</h3>
              <button
                onClick={() => {
                  setShowUploadModal(false);
                  setUploadFile(null);
                  setUploadError(null);
                  setUploadProgress(0);
                }}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4">
              {/* Instructions */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <h4 className="text-sm font-medium text-blue-900 mb-1">Upload Instructions</h4>
                <ul className="text-xs text-blue-800 space-y-1">
                  <li>• Upload JSON files with trace data (CSV support coming soon)</li>
                  <li>• Each trace should include: userInput, aiResponse, model info</li>
                  <li>• <span className="font-medium">Try the sample file:</span> <code className="bg-blue-100 px-1 rounded">sample_traces.json</code> in the project root</li>
                  <li>• Maximum file size: 10MB</li>
                </ul>
              </div>

              {/* File Upload Area */}
              <div 
                className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
                  uploadFile ? 'border-green-300 bg-green-50' : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50'
                }`}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".json,.csv,.xlsx"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                
                {uploadFile ? (
                  <div className="flex items-center justify-center gap-2 text-green-700">
                    <FileText className="w-6 h-6" />
                    <span className="font-medium">{uploadFile.name}</span>
                  </div>
                ) : (
                  <div className="text-gray-500">
                    <Upload className="w-8 h-8 mx-auto mb-2" />
                    <p className="font-medium">Choose a file or drag it here</p>
                    <p className="text-sm">Supports JSON, CSV, Excel files (max 10MB)</p>
                  </div>
                )}
              </div>

              {/* Error Message */}
              {uploadError && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                  <div className="flex items-center gap-2 text-red-700">
                    <AlertCircle className="w-4 h-4" />
                    <span className="text-sm">{uploadError}</span>
                  </div>
                </div>
              )}

              {/* Upload Progress */}
              {isUploading && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm text-gray-600">
                    <span>Uploading...</span>
                    <span>{uploadProgress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex gap-3 pt-2">
                <button
                  onClick={() => {
                    setShowUploadModal(false);
                    setUploadFile(null);
                    setUploadError(null);
                    setUploadProgress(0);
                  }}
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
                  disabled={isUploading}
                >
                  Cancel
                </button>
                <button
                  onClick={processUpload}
                  disabled={!uploadFile || isUploading}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isUploading ? 'Uploading...' : 'Upload'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Rejection Modal */}
      {showRejectionModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Reject Trace</h3>
              <button
                onClick={() => setShowRejectionModal(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Why are you rejecting this trace?
                </label>
                <div className="space-y-2">
                  {[
                    'Inaccurate response',
                    'Irrelevant to query',
                    'Poor quality',
                    'Hallucinated information',
                    'Inappropriate content',
                    'Technical error',
                    'Other'
                  ].map((reason) => (
                    <button
                      key={reason}
                      onClick={() => handleRejectionSubmit(reason)}
                      className="w-full text-left px-3 py-2 border border-gray-200 rounded-lg hover:bg-gray-50 text-sm"
                    >
                      {reason}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex gap-3 pt-2">
                <button
                  onClick={() => setShowRejectionModal(false)}
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
