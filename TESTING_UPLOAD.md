# Testing the Upload Data Feature

## Quick Test Instructions

1. **Start the application:**
   ```bash
   # Terminal 1: Start the backend
   cd backend
   python run_backend.py
   
   # Terminal 2: Start the frontend  
   cd frontend
   npm run dev
   ```

2. **Access the dashboard:**
   - Open http://localhost:3000 in your browser
   - You should see the main LLM Evaluation Dashboard

3. **Test the upload functionality:**
   - Click the blue "Upload Data" button in the top right
   - This will open the upload modal with instructions
   - Click "Choose a file or drag it here"
   - Select the `sample_traces.json` file from the project root
   - Click "Upload" to process the file
   - You should see 3 sample traces appear in the left panel

## Dashboard Differences Explained

### Main Dashboard (`/`)
- **Purpose**: Trace management and human evaluation interface
- **Features**: 
  - Upload and view trace data
  - Filter and search traces
  - Perform human evaluations (Accept/Reject)
  - Basic charts for agreement and acceptance rates
  - Download labeled data

### Analytics Dashboard (`/analytics`)  
- **Purpose**: Advanced analytics with detailed metrics and insights
- **Features**:
  - System health monitoring
  - Detailed performance metrics
  - User engagement analytics
  - Model comparison tables
  - Real-time updates toggle
  - Time-range filtering
  - Cost analysis

Think of the main dashboard as your daily workspace for evaluating traces, while the analytics dashboard provides executive-level insights and system monitoring.

## Sample Data Structure

The `sample_traces.json` file contains 3 example traces:

1. **Content Generator**: Blog writing about AI in healthcare
2. **Email Assistant**: Customer support response
3. **Code Assistant**: Python bug fixing

Each trace includes:
- Conversation (user input, AI response, system prompt)
- Function calls and execution details
- Metadata (model, costs, token counts, performance)
- Evaluation fields (status, scores)

## Troubleshooting

### Upload Not Working?
1. Check browser console (F12) for error messages
2. Ensure the file is valid JSON format
3. Verify file size is under 10MB
4. Try the provided `sample_traces.json` first

### No Traces Appearing?
1. Look for console log messages showing "Upload successful!"
2. Check that the file was parsed correctly
3. Verify the traces have the required fields

### Backend Connection Issues?
1. Ensure backend is running on port 8000
2. Check that CORS is properly configured
3. Verify no firewall blocking connections

## File Format Requirements

For custom JSON files, each trace should have this structure:

```json
{
  "id": "unique-trace-id",
  "timestamp": "2025-01-27T10:43:26Z",
  "tool": "Tool-Name",
  "scenario": "Use-Case-Name", 
  "conversation": {
    "userInput": "User's question or prompt",
    "aiResponse": "AI's response",
    "systemPrompt": "System instructions (optional)"
  },
  "metadata": {
    "modelName": "gpt-4",
    "latencyMs": 1200,
    "tokenCount": {"input": 28, "output": 89},
    "costUsd": 0.0045
  }
}
```

Required fields: `userInput`, `aiResponse`, `modelName`
Optional fields will use defaults if missing. 