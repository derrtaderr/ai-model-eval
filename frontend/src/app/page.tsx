'use client';

import { useState, useRef } from 'react';
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
  Code,
  Database,
  Zap,
  AlertCircle,
  FileText,
  X
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

interface FilterOptions {
  tools: string[];
  scenarios: string[];
  statuses: string[];
  dataSources: string[];
}

const mockTraces: Trace[] = [
  {
    id: 'trace-001',
    timestamp: '5/27/2025, 10:43:26 PM',
    tool: 'Listing-Finder',
    scenario: 'Multiple-Listings',
    status: 'pending',
    modelScore: 'pass',
    humanScore: null,
    dataSource: 'human',
    conversation: {
      systemPrompt: 'You are a helpful real estate assistant. Help users find properties that match their criteria.',
      userInput: 'Find me multiple listings in downtown area with 2+ bedrooms under $500k',
      aiResponse: 'I found 5 listings matching your criteria in the downtown area. Here are the top options:\n\n1. 123 Main St - 2 bed, 2 bath, $485k\n2. 456 Oak Ave - 3 bed, 2 bath, $495k\n3. 789 Pine St - 2 bed, 1.5 bath, $475k\n\nWould you like more details about any of these properties?'
    },
    functions: [
      {
        name: 'search_properties',
        parameters: { location: 'downtown', bedrooms_min: 2, price_max: 500000 },
        result: { found: 5, properties: ['123 Main St', '456 Oak Ave', '789 Pine St'] },
        executionTime: 145
      },
      {
        name: 'format_listings',
        parameters: { properties: ['123 Main St', '456 Oak Ave', '789 Pine St'] },
        result: { formatted_response: 'Property listings formatted' },
        executionTime: 23
      }
    ],
    metadata: {
      modelName: 'gpt-4-turbo',
      latencyMs: 1847,
      tokenCount: { input: 156, output: 203 },
      costUsd: 0.0087,
      temperature: 0.7,
      maxTokens: 1000
    }
  },
  {
    id: 'trace-002', 
    timestamp: '5/27/2025, 9:15:42 PM',
    tool: 'Email-Draft',
    scenario: 'Offer-Submission',
    status: 'accepted',
    modelScore: 'pass',
    humanScore: 'good',
    dataSource: 'human',
    conversation: {
      userInput: 'Submit offer for the property at 123 Main St',
      aiResponse: 'Offer submitted successfully',
    },
    functions: [],
    metadata: {
      modelName: 'Email-Draft',
      latencyMs: 50,
      tokenCount: { input: 5, output: 7 },
      costUsd: 0.002,
      temperature: 0.8,
      maxTokens: 50
    }
  },
  {
    id: 'trace-003',
    timestamp: '5/27/2025, 8:33:18 PM', 
    tool: 'Market-Analysis',
    scenario: 'Price-Trends',
    status: 'pending',
    modelScore: 'pass',
    humanScore: null,
    dataSource: 'synthetic',
    conversation: {
      userInput: 'What is the current price trend for properties in the city?',
      aiResponse: 'The price trend for properties in the city is stable',
    },
    functions: [],
    metadata: {
      modelName: 'Market-Analysis',
      latencyMs: 150,
      tokenCount: { input: 8, output: 12 },
      costUsd: 0.003,
      temperature: 0.6,
      maxTokens: 80
    }
  }
];

const filterOptions: FilterOptions = {
  tools: ['All Tools', 'Listing-Finder', 'Email-Draft', 'Market-Analysis'],
  scenarios: [
    'All Scenarios',
    'Multiple-Listings',
    'Offer-Submission', 
    'Comparative-Analysis',
    'Pet-Friendly-Listings',
    'Feedback-Request',
    'Rental-Yield',
    'Luxury-Properties',
    'Counter-Offer',
    'Price-Trends',
    'Investment-Properties'
  ],
  statuses: ['All Status', 'Pending', 'Accepted', 'Rejected'],
  dataSources: ['All Sources', 'Human', 'Synthetic']
};

// Empty chart data - no evaluations yet
const agreementData: { date: string; rate: number }[] = [];

const acceptanceData: { date: string; rate: number }[] = [];

export default function EvaluationDashboard() {
  const [selectedTrace, setSelectedTrace] = useState<Trace | null>(null);
  const [filters, setFilters] = useState({
    tool: 'All Tools',
    scenario: 'All Scenarios', 
    status: 'All Status',
    dataSource: 'All Sources'
  });
  const [activeTab, setActiveTab] = useState<'chat' | 'functions' | 'metadata'>('chat');
  
  // Data management state
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    const allowedTypes = ['application/json', 'text/csv', 'application/vnd.ms-excel'];
    const allowedExtensions = ['.json', '.csv', '.xlsx'];
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
  };

  const processUpload = async () => {
    if (!uploadFile) return;

    setIsUploading(true);
    setUploadProgress(0);

    // Simulate upload progress
    const progressInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return prev + 10;
      });
    }, 200);

    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // In real implementation, this would be an API call:
      // const formData = new FormData();
      // formData.append('file', uploadFile);
      // const response = await fetch('/api/upload-traces', { method: 'POST', body: formData });
      
      setUploadProgress(100);
      setTimeout(() => {
        setIsUploading(false);
        setShowUploadModal(false);
        setUploadFile(null);
        setUploadProgress(0);
        // Show success message or refresh data
      }, 500);
    } catch {
      setUploadError('Upload failed. Please try again.');
      setIsUploading(false);
      clearInterval(progressInterval);
    }
  };

  const downloadLabeledData = async () => {
    setIsDownloading(true);
    
    try {
      // Simulate API call to generate download
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // In real implementation, this would fetch from API:
      // const response = await fetch('/api/export-labeled-data');
      // const blob = await response.blob();
      
      // Generate mock CSV data
      const mockData = `trace_id,timestamp,tool,scenario,user_input,ai_response,human_evaluation,evaluator_notes
trace-001,2025-01-20T10:43:26Z,Listing-Finder,Multiple-Listings,"Find me multiple listings","I found 5 listings...",accepted,"Good response quality"
trace-002,2025-01-20T09:15:42Z,Email-Draft,Offer-Submission,"Submit offer","Offer submitted successfully",accepted,"Quick and accurate"
trace-003,2025-01-20T08:33:18Z,Market-Analysis,Price-Trends,"Price trend?","Price trend is stable",pending,""`;

      const blob = new Blob([mockData], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `labeled-data-${new Date().toISOString().split('T')[0]}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch {
      console.error('Download failed');
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

    return (
      <div className="bg-white p-6 rounded-lg border">
        <h3 className="text-sm font-medium text-gray-600 mb-4 flex items-center gap-2">
          {title}
          <div className="w-2 h-2 rounded-full bg-gray-400"></div>
        </h3>
        <div className="h-32 relative">
          <div className="absolute left-0 top-0 text-xs text-gray-700">100%</div>
          <div className="absolute left-0 top-1/2 text-xs text-gray-700">75%</div>
          <div className="absolute left-0 bottom-8 text-xs text-gray-700">50%</div>
          <div className="absolute left-0 bottom-0 text-xs text-gray-700">0%</div>
          
          <svg className="w-full h-full ml-6" viewBox="0 0 300 120">
            <polyline
              fill="none"
              stroke={color}
              strokeWidth="2"
              points={data.map((point, index) => 
                `${(index / (data.length - 1)) * 280 + 10},${120 - ((point.rate - minRate) / range) * 100}`
              ).join(' ')}
            />
            {data.map((point, index) => (
              <circle
                key={index}
                cx={(index / (data.length - 1)) * 280 + 10}
                cy={120 - ((point.rate - minRate) / range) * 100}
                r="3"
                fill={color}
              />
            ))}
          </svg>
          
          <div className="flex justify-between mt-2 ml-6 text-xs text-gray-700">
            {data.map(point => point.date).map((date, index) => (
              <span key={index}>{date}</span>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const FilterDropdown = ({ 
    value, 
    options, 
    onChange 
  }: { 
    value: string, 
    options: string[], 
    onChange: (value: string) => void 
  }) => {
    const [isOpen, setIsOpen] = useState(false);

    return (
      <div className="relative">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center gap-2 px-3 py-2 bg-white border rounded-lg hover:bg-gray-50 min-w-[140px] justify-between"
        >
          <span className="text-sm text-gray-900">{value}</span>
          <ChevronDown className="w-4 h-4 text-gray-600" />
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

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div>
              <h1 className="text-xl font-semibold text-gray-900">LLM Data Review</h1>
              <p className="text-sm text-gray-600">Review and label LLM interactions</p>
            </div>
            <div className="flex items-center gap-3">
              {/* Upload Data Button */}
              <button 
                className="bg-blue-600 text-white px-4 py-2 rounded-lg flex items-center gap-2 hover:bg-blue-700 transition-colors"
                onClick={() => setShowUploadModal(true)}
              >
                <Upload className="w-4 h-4" />
                Upload Data
              </button>

              {/* Download Labeled Data Button */}
              <button 
                className="bg-green-600 text-white px-4 py-2 rounded-lg flex items-center gap-2 hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                onClick={downloadLabeledData}
                disabled={isDownloading}
              >
                <Download className="w-4 h-4" />
                {isDownloading ? 'Downloading...' : 'Download Labeled Data'}
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-8">
          
          {/* Dashboard Header */}
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900">Dashboard</h2>
            <button className="text-sm text-blue-600 hover:text-blue-700">View Reports</button>
          </div>

          {/* Filter Section */}
          <div className="bg-white rounded-lg border p-6">
            <h3 className="text-sm font-medium text-gray-900 mb-4">Filter Records</h3>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <div>
                <label className="block text-xs text-gray-600 mb-1">Tool</label>
                <FilterDropdown
                  value={filters.tool}
                  options={filterOptions.tools}
                  onChange={(value) => setFilters({...filters, tool: value})}
                />
              </div>
              <div>
                <label className="block text-xs text-gray-600 mb-1">Scenario</label>
                <FilterDropdown
                  value={filters.scenario}
                  options={filterOptions.scenarios}
                  onChange={(value) => setFilters({...filters, scenario: value})}
                />
              </div>
              <div>
                <label className="block text-xs text-gray-600 mb-1">Status</label>
                <FilterDropdown
                  value={filters.status}
                  options={filterOptions.statuses}
                  onChange={(value) => setFilters({...filters, status: value})}
                />
              </div>
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

          {/* Analytics Charts - Now in the middle */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
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

          {/* Records List */}
          <div className="bg-white rounded-lg border">
            <div className="p-4 border-b">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">
                  Record 1 of 10
                </span>
                <div className="flex items-center gap-2">
                  <button className="text-sm text-blue-600 hover:text-blue-700">Chat</button>
                  <div className="flex items-center gap-2">
                    <button className="p-1 border rounded hover:bg-gray-50">
                      <ChevronLeft className="w-4 h-4" />
                    </button>
                    <span className="text-sm text-gray-600">Previous</span>
                    <span className="text-sm text-gray-600">Next</span>
                    <button className="p-1 border rounded hover:bg-gray-50">
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
                  {/* Trace Overview */}
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

                  {/* Tab Content */}
                  {activeTab === 'chat' && (
                    <div className="space-y-6">
                      {/* System Prompt */}
                      {selectedTrace.conversation.systemPrompt && (
                        <div className="bg-yellow-50 border border-yellow-200 p-4 rounded-lg">
                          <div className="flex items-center gap-2 text-sm text-yellow-800 font-medium mb-2">
                            <AlertCircle className="w-4 h-4" />
                            System Prompt
                          </div>
                          <div className="text-sm text-yellow-700">
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

                      {/* Evaluation Controls */}
                      <div className="bg-gray-50 border border-gray-200 p-4 rounded-lg">
                        <h4 className="text-sm font-medium text-gray-900 mb-3">Human Evaluation</h4>
                        <div className="flex gap-3">
                          <button className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700">
                            <CheckCircle className="w-4 h-4" />
                            Accept
                          </button>
                          <button className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700">
                            <XCircle className="w-4 h-4" />
                            Reject
                          </button>
                          <button className="flex items-center gap-2 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50">
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
                          />
                        </div>
                      </div>
                    </div>
                  )}

                  {activeTab === 'functions' && (
                    <div className="space-y-4">
                      {selectedTrace.functions.length > 0 ? (
                        selectedTrace.functions.map((func, index) => (
                          <div key={index} className="border border-gray-200 rounded-lg p-4">
                            <div className="flex items-center gap-2 mb-3">
                              <Code className="w-4 h-4 text-blue-600" />
                              <span className="font-medium text-gray-900">{func.name}</span>
                              <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                                {func.executionTime}ms
                              </span>
                            </div>
                            
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              <div>
                                <h5 className="text-sm font-medium text-gray-700 mb-2">Parameters</h5>
                                <pre className="text-xs bg-gray-50 p-2 rounded border overflow-x-auto text-gray-900">
                                  {JSON.stringify(func.parameters, null, 2)}
                                </pre>
                              </div>
                              <div>
                                <h5 className="text-sm font-medium text-gray-700 mb-2">Result</h5>
                                <pre className="text-xs bg-gray-50 p-2 rounded border overflow-x-auto text-gray-900">
                                  {JSON.stringify(func.result, null, 2)}
                                </pre>
                              </div>
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="text-center py-8 text-gray-600">
                          <Code className="w-8 h-8 mx-auto mb-2 text-gray-600" />
                          <div>No function calls in this trace</div>
                        </div>
                      )}
                    </div>
                  )}

                  {activeTab === 'metadata' && (
                    <div className="space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-4">
                          <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg">
                            <div className="flex items-center gap-2 text-sm text-blue-800 font-medium mb-2">
                              <Database className="w-4 h-4" />
                              Model Information
                            </div>
                            <div className="space-y-2 text-sm">
                              <div>
                                <span className="text-gray-600">Model:</span>
                                <span className="ml-2 text-gray-900">{selectedTrace.metadata.modelName}</span>
                              </div>
                              <div>
                                <span className="text-gray-600">Temperature:</span>
                                <span className="ml-2 text-gray-900">{selectedTrace.metadata.temperature}</span>
                              </div>
                              <div>
                                <span className="text-gray-600">Max Tokens:</span>
                                <span className="ml-2 text-gray-900">{selectedTrace.metadata.maxTokens}</span>
                              </div>
                            </div>
                          </div>

                          <div className="bg-green-50 border border-green-200 p-4 rounded-lg">
                            <div className="flex items-center gap-2 text-sm text-green-800 font-medium mb-2">
                              <Zap className="w-4 h-4" />
                              Performance Metrics
                            </div>
                            <div className="space-y-2 text-sm">
                              <div>
                                <span className="text-gray-600">Latency:</span>
                                <span className="ml-2 text-gray-900">{selectedTrace.metadata.latencyMs}ms</span>
                              </div>
                              <div>
                                <span className="text-gray-600">Input Tokens:</span>
                                <span className="ml-2 text-gray-900">{selectedTrace.metadata.tokenCount.input}</span>
                              </div>
                              <div>
                                <span className="text-gray-600">Output Tokens:</span>
                                <span className="ml-2 text-gray-900">{selectedTrace.metadata.tokenCount.output}</span>
                              </div>
                              <div>
                                <span className="text-gray-600">Cost:</span>
                                <span className="ml-2 text-gray-900">${selectedTrace.metadata.costUsd.toFixed(4)}</span>
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="bg-gray-50 border border-gray-200 p-4 rounded-lg">
                          <h5 className="text-sm font-medium text-gray-700 mb-3">Raw Metadata</h5>
                          <pre className="text-xs bg-white p-3 rounded border overflow-x-auto text-gray-900">
                            {JSON.stringify(selectedTrace.metadata, null, 2)}
                          </pre>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8">
                  <div className="text-gray-500">Select a trace record to view details</div>
                  <button
                    onClick={() => setSelectedTrace(mockTraces[0])}
                    className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                  >
                    Load Sample Trace
                  </button>
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
    </div>
  );
}
