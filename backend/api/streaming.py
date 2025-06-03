from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional, AsyncGenerator
import asyncio
import json
import time
from datetime import datetime

from ..database.connection import get_db_connection
from ..services.cache_service import cache_service

router = APIRouter()

class StreamManager:
    """Manages active SSE connections"""
    def __init__(self):
        self.connections: Dict[str, AsyncGenerator] = {}
    
    async def add_connection(self, client_id: str, generator: AsyncGenerator):
        """Add a new SSE connection"""
        self.connections[client_id] = generator
    
    async def remove_connection(self, client_id: str):
        """Remove an SSE connection"""
        if client_id in self.connections:
            del self.connections[client_id]
    
    async def broadcast(self, event: str, data: Dict[str, Any]):
        """Broadcast an event to all connected clients"""
        disconnected = []
        for client_id, generator in self.connections.items():
            try:
                await generator.asend({
                    "event": event,
                    "data": data,
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception:
                disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            await self.remove_connection(client_id)

stream_manager = StreamManager()

async def create_sse_stream(client_id: str, filters: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
    """Create a Server-Sent Events stream for real-time updates"""
    try:
        # Send initial connection event
        yield f"data: {json.dumps({'event': 'connected', 'client_id': client_id, 'timestamp': datetime.utcnow().isoformat()})}\n\n"
        
        # Subscribe to Redis pub/sub for real-time updates
        pubsub = await cache_service.get_pubsub()
        await pubsub.subscribe("trace_updates", "evaluation_updates", "system_updates")
        
        while True:
            try:
                # Check for new messages
                message = await pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    data = json.loads(message['data'])
                    
                    # Apply filters if specified
                    if filters and not _matches_filters(data, filters):
                        continue
                    
                    # Format SSE message
                    sse_data = {
                        "event": message['channel'],
                        "data": data,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    yield f"data: {json.dumps(sse_data)}\n\n"
                
                # Send heartbeat every 30 seconds
                if int(time.time()) % 30 == 0:
                    heartbeat = {
                        "event": "heartbeat",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    yield f"data: {json.dumps(heartbeat)}\n\n"
                
                await asyncio.sleep(0.1)  # Small delay to prevent CPU spinning
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                error_msg = {
                    "event": "error",
                    "data": {"message": str(e)},
                    "timestamp": datetime.utcnow().isoformat()
                }
                yield f"data: {json.dumps(error_msg)}\n\n"
                break
        
    except Exception as e:
        error_msg = {
            "event": "connection_error", 
            "data": {"message": str(e)},
            "timestamp": datetime.utcnow().isoformat()
        }
        yield f"data: {json.dumps(error_msg)}\n\n"
    
    finally:
        # Clean up
        try:
            await pubsub.unsubscribe()
            await pubsub.close()
        except:
            pass

def _matches_filters(data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Check if data matches the specified filters"""
    for key, value in filters.items():
        if key in data and data[key] != value:
            return False
    return True

@router.get("/stream/events")
async def stream_events(
    request: Request,
    client_id: Optional[str] = None,
    model_name: Optional[str] = None,
    evaluation_status: Optional[str] = None
):
    """Stream real-time events via Server-Sent Events"""
    if not client_id:
        client_id = f"client_{int(time.time() * 1000)}"
    
    # Build filters
    filters = {}
    if model_name:
        filters['model_name'] = model_name
    if evaluation_status:
        filters['evaluation_status'] = evaluation_status
    
    async def event_stream():
        async for data in create_sse_stream(client_id, filters):
            yield data
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@router.get("/stream/traces")
async def stream_traces(
    request: Request,
    client_id: Optional[str] = None,
    limit: int = 10
):
    """Stream recent traces with real-time updates"""
    if not client_id:
        client_id = f"trace_client_{int(time.time() * 1000)}"
    
    async def trace_stream():
        try:
            # Send initial recent traces
            db_connection = next(get_db_connection())
            cursor = db_connection.cursor()
            
            cursor.execute("""
                SELECT id, timestamp, model_name, user_query, ai_response, 
                       evaluation_status, tokens_used, response_time_ms, cost
                FROM traces 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            recent_traces = []
            for row in cursor.fetchall():
                recent_traces.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "model_name": row[2],
                    "user_query": row[3][:100] + "..." if len(row[3]) > 100 else row[3],
                    "ai_response": row[4][:100] + "..." if len(row[4]) > 100 else row[4],
                    "evaluation_status": row[5],
                    "tokens_used": row[6],
                    "response_time_ms": row[7],
                    "cost": row[8]
                })
            
            initial_data = {
                "event": "initial_traces",
                "data": recent_traces,
                "timestamp": datetime.utcnow().isoformat()
            }
            yield f"data: {json.dumps(initial_data)}\n\n"
            
            # Stream real-time updates
            async for data in create_sse_stream(client_id, {"event_type": "trace"}):
                yield data
                
        except Exception as e:
            error_msg = {
                "event": "error",
                "data": {"message": str(e)},
                "timestamp": datetime.utcnow().isoformat()
            }
            yield f"data: {json.dumps(error_msg)}\n\n"
    
    return StreamingResponse(
        trace_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@router.get("/stream/metrics")
async def stream_metrics(
    request: Request,
    client_id: Optional[str] = None,
    interval: int = 5  # seconds
):
    """Stream real-time metrics and statistics"""
    if not client_id:
        client_id = f"metrics_client_{int(time.time() * 1000)}"
    
    async def metrics_stream():
        try:
            last_update = 0
            
            while True:
                current_time = time.time()
                
                # Update metrics every interval
                if current_time - last_update >= interval:
                    try:
                        db_connection = next(get_db_connection())
                        cursor = db_connection.cursor()
                        
                        # Get current metrics
                        cursor.execute("""
                            SELECT 
                                COUNT(*) as total_traces,
                                COUNT(CASE WHEN evaluation_status = 'accepted' THEN 1 END) as accepted_traces,
                                COUNT(CASE WHEN evaluation_status = 'rejected' THEN 1 END) as rejected_traces,
                                COUNT(CASE WHEN evaluation_status = 'pending' THEN 1 END) as pending_traces,
                                AVG(CASE WHEN response_time_ms IS NOT NULL THEN response_time_ms END) as avg_response_time,
                                SUM(CASE WHEN cost IS NOT NULL THEN cost END) as total_cost,
                                COUNT(CASE WHEN DATE(timestamp) = DATE('now') THEN 1 END) as today_traces
                            FROM traces
                        """)
                        
                        row = cursor.fetchone()
                        
                        metrics = {
                            "total_traces": row[0] or 0,
                            "accepted_traces": row[1] or 0,
                            "rejected_traces": row[2] or 0,
                            "pending_traces": row[3] or 0,
                            "avg_response_time": round(row[4], 2) if row[4] else 0,
                            "total_cost": round(row[5], 4) if row[5] else 0,
                            "today_traces": row[6] or 0,
                            "acceptance_rate": round((row[1] or 0) / max(row[0] or 1, 1) * 100, 1)
                        }
                        
                        metrics_data = {
                            "event": "metrics_update",
                            "data": metrics,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
                        yield f"data: {json.dumps(metrics_data)}\n\n"
                        last_update = current_time
                        
                    except Exception as e:
                        error_msg = {
                            "event": "metrics_error",
                            "data": {"message": str(e)},
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        yield f"data: {json.dumps(error_msg)}\n\n"
                
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            error_msg = {
                "event": "stream_error",
                "data": {"message": str(e)},
                "timestamp": datetime.utcnow().isoformat()
            }
            yield f"data: {json.dumps(error_msg)}\n\n"
    
    return StreamingResponse(
        metrics_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@router.post("/stream/broadcast")
async def broadcast_event(
    event: str,
    data: Dict[str, Any],
    target_clients: Optional[list] = None
):
    """Manually broadcast an event to connected clients"""
    try:
        await stream_manager.broadcast(event, data)
        return {
            "success": True,
            "message": f"Event '{event}' broadcasted",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to broadcast: {str(e)}") 