"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { toast } from "@/components/ui/use-toast";
import { Loader2, CheckCircle, XCircle, AlertCircle, Database, DollarSign, Clock, Settings, TestTube, RefreshCw, Play, TrendingUp } from 'lucide-react';

interface Provider {
  type: string;
  name: string;
  is_available: boolean;
  is_healthy: boolean;
  last_check: string | null;
  response_time_ms: number | null;
  error_message: string | null;
  requests_count: number;
  total_cost_usd: number;
  avg_response_time_ms: number;
}

interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  capabilities: string[];
  context_window: number | null;
  cost_per_token_input: number | null;
  cost_per_token_output: number | null;
  supports_streaming: boolean;
  supports_functions: boolean;
  supports_vision: boolean;
}

interface ProviderHealth {
  is_healthy: boolean;
  last_check: string;
  response_time_ms: number | null;
  error_message: string | null;
  available_models: string[];
  rate_limit_status: Record<string, unknown> | null;
}

interface UsageStats {
  requests_count: number;
  tokens_input: number;
  tokens_output: number;
  total_cost_usd: number;
  avg_response_time_ms: number;
  error_count: number;
  last_request: string | null;
  cost_per_request: number;
  success_rate: number;
}

interface ProvidersOverview {
  providers: Provider[];
  summary: {
    total_providers: number;
    available_providers: number;
    total_requests: number;
    total_cost_usd: number;
    last_updated: string;
  };
}

interface CompletionRequest {
  prompt: string;
  provider: string;
  model: string;
  max_tokens: number;
  temperature: number;
}

interface CostAnalysis {
  total_cost_usd: number;
  average_daily_cost: number;
  projected_monthly_cost: number;
  cost_breakdown: Array<{
    provider: string;
    cost_usd: number;
    percentage: number;
  }>;
  recommendations: string[];
}

interface TestResult {
  status: string;
  response_content?: string;
  response_time_ms?: number;
  cost_usd?: number;
  model_used?: string;
  error_message?: string;
}

interface CompletionResult {
  provider: string;
  model: string;
  content: string;
  response_time_ms?: number;
  cost_usd?: number;
}

export default function ProvidersPage() {
  const [overview, setOverview] = useState<ProvidersOverview | null>(null);
  const [models, setModels] = useState<Record<string, ModelInfo[]>>({});
  const [usageStats, setUsageStats] = useState<Record<string, UsageStats>>({});
  const [costAnalysis, setCostAnalysis] = useState<CostAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [testDialogOpen, setTestDialogOpen] = useState(false);
  const [completionDialogOpen, setCompletionDialogOpen] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string>('');
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [completionResult, setCompletionResult] = useState<CompletionResult | null>(null);
  const [testing, setTesting] = useState(false);
  const [completing, setCompleting] = useState(false);

  // Test form state
  const [testPrompt, setTestPrompt] = useState("Hello! Please respond with a brief greeting.");
  
  // Completion form state
  const [completionRequest, setCompletionRequest] = useState<CompletionRequest>({
    prompt: "What are the key benefits of using AI in software development?",
    provider: '',
    model: '',
    max_tokens: 150,
    temperature: 0.7
  });

  const fetchData = async () => {
    try {
      const [overviewRes, modelsRes, statsRes, costRes] = await Promise.all([
        fetch('/api/v1/llm-providers/'),
        fetch('/api/v1/llm-providers/models'),
        fetch('/api/v1/llm-providers/usage-stats'),
        fetch('/api/v1/llm-providers/cost-analysis?days=7')
      ]);

      if (overviewRes.ok) {
        setOverview(await overviewRes.json());
      }
      if (modelsRes.ok) {
        setModels(await modelsRes.json());
      }
      if (statsRes.ok) {
        setUsageStats(await statsRes.json());
      }
      if (costRes.ok) {
        setCostAnalysis(await costRes.json());
      }
    } catch (error) {
      console.error('Error fetching providers data:', error);
      toast({
        title: "Error",
        description: "Failed to load provider data",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    await fetchData();
  };

  const handleTest = async (provider: string) => {
    setTesting(true);
    try {
      const response = await fetch('/api/v1/llm-providers/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: provider,
          test_prompt: testPrompt
        })
      });

      const result = await response.json();
      setTestResult(result);
      
      if (result.status === 'success') {
        toast({
          title: "Test Successful",
          description: `${provider} responded successfully`,
        });
      } else {
        toast({
          title: "Test Failed",
          description: result.error_message || "Test failed",
          variant: "destructive",
        });
      }
    } catch {
      toast({
        title: "Error",
        description: "Failed to test provider",
        variant: "destructive",
      });
    } finally {
      setTesting(false);
    }
  };

  const handleCompletion = async () => {
    setCompleting(true);
    try {
      const response = await fetch('/api/v1/llm-providers/complete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: completionRequest.prompt,
          provider: completionRequest.provider,
          model: completionRequest.model,
          max_tokens: completionRequest.max_tokens,
          temperature: completionRequest.temperature,
          operation_type: "chat"
        })
      });

      if (response.ok) {
        const result = await response.json();
        setCompletionResult(result);
        toast({
          title: "Completion Successful",
          description: `Generated response from ${result.provider}`,
        });
      } else {
        throw new Error('Request failed');
      }
    } catch {
      toast({
        title: "Error",
        description: "Failed to complete request",
        variant: "destructive",
      });
    } finally {
      setCompleting(false);
    }
  };

  const getProviderStatusIcon = (provider: Provider) => {
    if (provider.is_healthy) {
      return <CheckCircle className="h-5 w-5 text-green-500" />;
    } else if (provider.is_available) {
      return <AlertCircle className="h-5 w-5 text-yellow-500" />;
    } else {
      return <XCircle className="h-5 w-5 text-red-500" />;
    }
  };

  const getProviderStatusText = (provider: Provider) => {
    if (provider.is_healthy) return "Healthy";
    if (provider.is_available) return "Available";
    return "Unavailable";
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 4
    }).format(amount);
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('en-US').format(num);
  };

  const getCapabilityBadgeColor = (capability: string) => {
    const colors: Record<string, string> = {
      'text_generation': 'bg-blue-100 text-blue-800',
      'chat': 'bg-green-100 text-green-800',
      'reasoning': 'bg-purple-100 text-purple-800',
      'code_generation': 'bg-orange-100 text-orange-800',
      'vision': 'bg-pink-100 text-pink-800',
      'function_calling': 'bg-indigo-100 text-indigo-800',
      'embeddings': 'bg-gray-100 text-gray-800',
      'multimodal': 'bg-red-100 text-red-800'
    };
    return colors[capability] || 'bg-gray-100 text-gray-800';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading providers...</span>
      </div>
    );
  }

  return (
    <div className="container mx-auto py-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">LLM Providers</h1>
          <p className="text-muted-foreground mt-1">
            Manage OpenAI, Anthropic, and other LLM provider integrations
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={handleRefresh}
            variant="outline"
            disabled={refreshing}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Dialog open={completionDialogOpen} onOpenChange={setCompletionDialogOpen}>
            <DialogTrigger asChild>
              <Button>
                <Play className="h-4 w-4 mr-2" />
                Test Completion
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl">
              <DialogHeader>
                <DialogTitle>Test LLM Completion</DialogTitle>
                <DialogDescription>
                  Test a completion request with any available provider and model.
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="provider">Provider</Label>
                    <Select 
                      value={completionRequest.provider} 
                      onValueChange={(value) => setCompletionRequest(prev => ({ ...prev, provider: value, model: '' }))}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select provider" />
                      </SelectTrigger>
                      <SelectContent>
                        {overview?.providers.filter(p => p.is_healthy).map(provider => (
                          <SelectItem key={provider.type} value={provider.type}>
                            {provider.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="model">Model</Label>
                    <Select 
                      value={completionRequest.model} 
                      onValueChange={(value) => setCompletionRequest(prev => ({ ...prev, model: value }))}
                      disabled={!completionRequest.provider}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select model" />
                      </SelectTrigger>
                      <SelectContent>
                        {models[completionRequest.provider]?.map(model => (
                          <SelectItem key={model.id} value={model.id}>
                            {model.name} ({model.id})
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="max_tokens">Max Tokens</Label>
                    <Input
                      type="number"
                      value={completionRequest.max_tokens}
                      onChange={(e) => setCompletionRequest(prev => ({ ...prev, max_tokens: parseInt(e.target.value) || 150 }))}
                      min={1}
                      max={4000}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="temperature">Temperature</Label>
                    <Input
                      type="number"
                      step="0.1"
                      value={completionRequest.temperature}
                      onChange={(e) => setCompletionRequest(prev => ({ ...prev, temperature: parseFloat(e.target.value) || 0.7 }))}
                      min={0}
                      max={2}
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="prompt">Prompt</Label>
                  <Textarea
                    value={completionRequest.prompt}
                    onChange={(e) => setCompletionRequest(prev => ({ ...prev, prompt: e.target.value }))}
                    rows={4}
                  />
                </div>
                {completionResult && (
                  <div className="border rounded-lg p-4 bg-gray-50">
                    <h4 className="font-semibold mb-2">Response:</h4>
                    <p className="text-sm mb-2">{completionResult.content}</p>
                    <div className="text-xs text-muted-foreground space-y-1">
                      <div>Provider: {completionResult.provider}</div>
                      <div>Model: {completionResult.model}</div>
                      <div>Response Time: {completionResult.response_time_ms?.toFixed(1)}ms</div>
                      <div>Cost: {formatCurrency(completionResult.cost_usd || 0)}</div>
                    </div>
                  </div>
                )}
              </div>
              <DialogFooter>
                <Button 
                  onClick={handleCompletion}
                  disabled={completing || !completionRequest.provider || !completionRequest.model}
                >
                  {completing && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                  {completing ? 'Generating...' : 'Generate Response'}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Summary Cards */}
      {overview && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Providers</CardTitle>
              <Database className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{overview.summary.total_providers}</div>
              <p className="text-xs text-muted-foreground">
                {overview.summary.available_providers} available
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Requests</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatNumber(overview.summary.total_requests)}</div>
              <p className="text-xs text-muted-foreground">
                Across all providers
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Cost</CardTitle>
              <DollarSign className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatCurrency(overview.summary.total_cost_usd)}</div>
              <p className="text-xs text-muted-foreground">
                All-time usage
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Last Updated</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-lg font-bold">
                {new Date(overview.summary.last_updated).toLocaleTimeString()}
              </div>
              <p className="text-xs text-muted-foreground">
                System status
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      <Tabs defaultValue="providers" className="space-y-4">
        <TabsList>
          <TabsTrigger value="providers">Providers</TabsTrigger>
          <TabsTrigger value="models">Models</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="costs">Cost Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="providers" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {overview?.providers.map((provider) => (
              <Card key={provider.type}>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>{provider.name}</span>
                    {getProviderStatusIcon(provider)}
                  </CardTitle>
                  <CardDescription>
                    Status: {getProviderStatusText(provider)}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Requests:</span>
                      <span className="font-medium">{formatNumber(provider.requests_count)}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Total Cost:</span>
                      <span className="font-medium">{formatCurrency(provider.total_cost_usd)}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Avg Response:</span>
                      <span className="font-medium">
                        {provider.avg_response_time_ms ? `${provider.avg_response_time_ms.toFixed(1)}ms` : 'N/A'}
                      </span>
                    </div>
                    {provider.error_message && (
                      <div className="text-xs text-red-600 bg-red-50 p-2 rounded">
                        {provider.error_message}
                      </div>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <Dialog open={testDialogOpen && selectedProvider === provider.type} onOpenChange={(open) => {
                      setTestDialogOpen(open);
                      if (open) setSelectedProvider(provider.type);
                    }}>
                      <DialogTrigger asChild>
                        <Button 
                          variant="outline" 
                          size="sm"
                          disabled={!provider.is_healthy}
                          className="flex-1"
                        >
                          <TestTube className="h-4 w-4 mr-1" />
                          Test
                        </Button>
                      </DialogTrigger>
                      <DialogContent>
                        <DialogHeader>
                          <DialogTitle>Test {provider.name}</DialogTitle>
                          <DialogDescription>
                            Send a test request to verify the provider is working correctly.
                          </DialogDescription>
                        </DialogHeader>
                        <div className="space-y-4">
                          <div className="space-y-2">
                            <Label htmlFor="test-prompt">Test Prompt</Label>
                            <Textarea
                              id="test-prompt"
                              value={testPrompt}
                              onChange={(e) => setTestPrompt(e.target.value)}
                              rows={3}
                            />
                          </div>
                          {testResult && (
                            <div className="border rounded-lg p-4 bg-gray-50">
                              <h4 className="font-semibold mb-2">Test Result:</h4>
                              {testResult.status === 'success' ? (
                                <div className="space-y-2">
                                  <p className="text-sm text-green-600">✅ Test successful</p>
                                  <p className="text-sm"><strong>Response:</strong> {testResult.response_content}</p>
                                  <p className="text-xs text-muted-foreground">
                                    Response time: {testResult.response_time_ms?.toFixed(1)}ms | 
                                    Cost: {formatCurrency(testResult.cost_usd || 0)} | 
                                    Model: {testResult.model_used}
                                  </p>
                                </div>
                              ) : (
                                <div>
                                  <p className="text-sm text-red-600">❌ Test failed</p>
                                  <p className="text-xs text-muted-foreground">{testResult.error_message}</p>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                        <DialogFooter>
                          <Button 
                            onClick={() => handleTest(provider.type)}
                            disabled={testing}
                          >
                            {testing && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                            {testing ? 'Testing...' : 'Run Test'}
                          </Button>
                        </DialogFooter>
                      </DialogContent>
                    </Dialog>
                    <Button variant="outline" size="sm" className="flex-1">
                      <Settings className="h-4 w-4 mr-1" />
                      Config
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="models" className="space-y-4">
          <div className="space-y-6">
            {Object.entries(models).map(([providerType, modelList]) => (
              <Card key={providerType}>
                <CardHeader>
                  <CardTitle className="capitalize">{providerType} Models</CardTitle>
                  <CardDescription>
                    {modelList.length} models available
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {modelList.map((model) => (
                      <div key={model.id} className="border rounded-lg p-4 space-y-3">
                        <div>
                          <h4 className="font-semibold">{model.name}</h4>
                          <p className="text-sm text-muted-foreground">{model.id}</p>
                        </div>
                        <div className="space-y-2">
                          <div className="flex flex-wrap gap-1">
                            {model.capabilities.map((capability) => (
                              <Badge 
                                key={capability} 
                                variant="secondary"
                                className={`text-xs ${getCapabilityBadgeColor(capability)}`}
                              >
                                {capability.replace('_', ' ')}
                              </Badge>
                            ))}
                          </div>
                          <div className="text-xs space-y-1">
                            {model.context_window && (
                              <div>Context: {formatNumber(model.context_window)} tokens</div>
                            )}
                            {model.cost_per_token_input && (
                              <div>
                                Cost: ${(model.cost_per_token_input * 1000).toFixed(4)}/1K input, 
                                ${(model.cost_per_token_output! * 1000).toFixed(4)}/1K output
                              </div>
                            )}
                            <div className="flex gap-2">
                              {model.supports_streaming && <Badge variant="outline" className="text-xs">Streaming</Badge>}
                              {model.supports_functions && <Badge variant="outline" className="text-xs">Functions</Badge>}
                              {model.supports_vision && <Badge variant="outline" className="text-xs">Vision</Badge>}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(usageStats).map(([providerType, stats]) => (
              <Card key={providerType}>
                <CardHeader>
                  <CardTitle className="capitalize">{providerType} Usage Stats</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-2xl font-bold">{formatNumber(stats.requests_count)}</div>
                      <p className="text-xs text-muted-foreground">Total Requests</p>
                    </div>
                    <div>
                      <div className="text-2xl font-bold">{stats.success_rate.toFixed(1)}%</div>
                      <p className="text-xs text-muted-foreground">Success Rate</p>
                    </div>
                    <div>
                      <div className="text-2xl font-bold">{formatCurrency(stats.total_cost_usd)}</div>
                      <p className="text-xs text-muted-foreground">Total Cost</p>
                    </div>
                    <div>
                      <div className="text-2xl font-bold">{stats.avg_response_time_ms.toFixed(1)}ms</div>
                      <p className="text-xs text-muted-foreground">Avg Response</p>
                    </div>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Input Tokens:</span>
                      <span className="font-medium">{formatNumber(stats.tokens_input)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Output Tokens:</span>
                      <span className="font-medium">{formatNumber(stats.tokens_output)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Cost per Request:</span>
                      <span className="font-medium">{formatCurrency(stats.cost_per_request)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Error Count:</span>
                      <span className="font-medium">{stats.error_count}</span>
                    </div>
                    {stats.last_request && (
                      <div className="flex justify-between">
                        <span>Last Request:</span>
                        <span className="font-medium">
                          {new Date(stats.last_request).toLocaleString()}
                        </span>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="costs" className="space-y-4">
          {costAnalysis && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Cost Summary (Last 7 Days)</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>Total Cost:</span>
                      <span className="font-bold text-lg">{formatCurrency(costAnalysis.total_cost_usd)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Daily Average:</span>
                      <span className="font-medium">{formatCurrency(costAnalysis.average_daily_cost)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Projected Monthly:</span>
                      <span className="font-medium">{formatCurrency(costAnalysis.projected_monthly_cost)}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Cost Breakdown</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {costAnalysis.cost_breakdown.map((item: any) => (
                      <div key={item.provider} className="space-y-2">
                        <div className="flex justify-between">
                          <span className="capitalize">{item.provider}</span>
                          <span className="font-medium">{formatCurrency(item.cost_usd)}</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-blue-600 h-2 rounded-full" 
                            style={{ width: `${item.percentage}%` }}
                          ></div>
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {item.percentage.toFixed(1)}% of total costs
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card className="md:col-span-2">
                <CardHeader>
                  <CardTitle>Recommendations</CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {costAnalysis.recommendations.map((rec: string, index: number) => (
                      <li key={index} className="flex items-start gap-2">
                        <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                        <span className="text-sm">{rec}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
} 