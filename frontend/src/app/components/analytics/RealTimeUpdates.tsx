import { useEffect, useState } from 'react';
import { RealTimeUpdate } from '../../types/analytics';

interface RealTimeUpdatesProps {
  onUpdate: (update: RealTimeUpdate) => void;
}

export function RealTimeUpdates({ onUpdate }: RealTimeUpdatesProps) {
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);

  useEffect(() => {
    // WebSocket connection for real-time updates
    // This would connect to the backend WebSocket endpoint
    // For now, we'll simulate with a mock connection
    
    const mockUpdates = () => {
      // Simulate real-time updates
      setIsConnected(true);
      
      const interval = setInterval(() => {
        const mockUpdate: RealTimeUpdate = {
          type: 'metric_update',
          data: {
            metric: 'system_performance',
            value: Math.random() * 100,
            timestamp: new Date().toISOString()
          },
          level: 'info',
          title: 'Performance Update',
          message: 'System metrics updated',
          timestamp: new Date().toISOString()
        };
        
        onUpdate(mockUpdate);
        setLastUpdate(new Date().toLocaleTimeString());
      }, 30000); // Update every 30 seconds
      
      return () => {
        clearInterval(interval);
        setIsConnected(false);
      };
    };

    const cleanup = mockUpdates();
    return cleanup;
  }, [onUpdate]);

  // Real WebSocket implementation would look like this:
  /*
  useEffect(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws/dashboard';
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      setIsConnected(true);
      console.log('Connected to real-time updates');
    };
    
    ws.onmessage = (event) => {
      try {
        const update: RealTimeUpdate = JSON.parse(event.data);
        onUpdate(update);
        setLastUpdate(new Date().toLocaleTimeString());
      } catch (error) {
        console.error('Failed to parse real-time update:', error);
      }
    };
    
    ws.onclose = () => {
      setIsConnected(false);
      console.log('Disconnected from real-time updates');
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };
    
    return () => {
      ws.close();
    };
  }, [onUpdate]);
  */

  return (
    <div className="fixed bottom-4 right-4 bg-white border border-gray-200 rounded-lg shadow-lg p-3 max-w-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <span className="text-sm font-medium text-gray-900">
            Real-time Updates
          </span>
        </div>
        <div className="text-xs text-gray-500">
          {isConnected ? 'Connected' : 'Disconnected'}
        </div>
      </div>
      
      {lastUpdate && (
        <div className="mt-2 text-xs text-gray-600">
          Last update: {lastUpdate}
        </div>
      )}
    </div>
  );
} 