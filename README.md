# BlueBubbles MCP Server

A Model Context Protocol (MCP) server that provides Claude Desktop with the ability to interact with a BlueBubbles instance for iMessage management.

## Overview

This MCP server enables Claude to:
- Search and retrieve iMessages
- Send messages to contacts or phone numbers
- Manage chat conversations
- Access contact information
- Monitor server status

## Features

### Message Management
- **Search Messages**: Find messages across all chats or within specific conversations
- **Recent Messages**: Retrieve the latest messages from any chat
- **Send Messages**: Send new messages to existing chats or phone numbers

### Chat Operations
- **List Chats**: Browse all available conversations with participants
- **Chat Details**: Get detailed information about specific conversations
- **Mark as Read**: Mark conversations as read

### Contact Management
- **Contact List**: Retrieve contacts with phone numbers and email addresses
- **Contact Search**: Find specific contacts in your address book

### Server Information
- **Server Status**: Check BlueBubbles server health and configuration
- **API Availability**: Verify connection and API version

## Prerequisites

- [BlueBubbles Server](https://bluebubbles.app/) running on macOS
- Python 3.11 or higher
- Access to BlueBubbles API with password authentication

## Installation

### Option 1: Docker Desktop (Recommended)

1. Clone or download this repository
2. Build the Docker image:
   ```bash
   docker build -t bluebubbles-mcp-server .
   ```

3. Run the container:
   ```bash
   docker run -e BLUEBUBBLES_URL=http://your-server:1234 \
              -e BLUEBUBBLES_PASSWORD=your-password \
              bluebubbles-mcp-server
   ```

4. Set Up Secrets

   ```docker mcp secret set BLUEBUBBLES_URL="http://your-bluebubbles-server:port"

   docker mcp secret set BLUEBUBBLES_PASSWORD="your-password"

   docker mcp secret ls
   ```

5. Create Custom Catolog (if it doesn't exist)

   <!-- Create catalogs directory if it doesn't exist -->
   `mkdir -p ~/.docker/mcp/catalogs`

   <!-- Create or edit custom.yaml - Can edit via editor of your choice -->
   `nano ~/.docker/mcp/catalogs/custom.yaml`

   <!-- Example custom.yaml file -->
   ```version: 2
   name: custom
   displayName: Custom MCP Servers
   registry:
   bluebubbles:
      description: "Interface for BlueBubbles iMessage API"
      title: "BlueBubbles"
      type: server
      dateAdded: "2025-09-26T17:20:42Z"
      image: bluebubbles-mcp-server:latest
      ref: ""
      readme: ""
      toolsUrl: ""
      source: ""
      upstream: ""
      icon: ""
      tools:
         - name: search_messages
         - name: get_recent_messages
         - name: list_chats
         - name: send_message
         - name: send_message_to_number
         - name: get_contacts
         - name: mark_chat_read
         - name: get_server_info
         - name: get_chat_details
      secrets:
         - name: BLUEBUBBLES_URL
         env: BLUEBUBBLES_URL
         example: "http://192.168.1.100:1234"
         - name: BLUEBUBBLES_PASSWORD
         env: BLUEBUBBLES_PASSWORD
         example: "your-password-here"
      metadata:
         category: integration
         tags:
         - messaging
         - imessage
         - communication
         - bluebubbles
         license: MIT
         owner: local
   ```

6. Update registry.yaml

   `nano ~/.docker/mcp/registry.yaml`

   ```registry:
      <!-- ... existing servers ... -->
      bluebubbles:
         ref: ""
   ```

7. Update Claude Desktop Config

   ```{
      "mcpServers": {
         "mcp-toolkit-gateway": {
               "command": "docker",
               "args": [
                  "run",
                  "-i",
                  "--rm",
                  "-v",
                  "/var/run/docker.sock:/var/run/docker.sock",
                  "-v",
                  "C:/Users/YOUR USER/.docker/mcp:/mcp", <!-- This will vary between Windows/Linux/Mac -->
                  "docker/mcp-gateway",
                  "--catalog=/mcp/catalogs/docker-mcp.yaml",
                  "--catalog=/mcp/catalogs/custom.yaml",
                  "--config=/mcp/config.yaml",
                  "--registry=/mcp/registry.yaml",
                  "--tools-config=/mcp/tools.yaml",
                  "--transport=stdio"
               ]
         }
      }
   }```

### Option 2: Local Installation (Not Tested)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   export BLUEBUBBLES_URL=http://your-server:1234
   export BLUEBUBBLES_PASSWORD=your-password
   ```

3. Run the server:
   ```bash
   python bluebubbles_server.py
   ```

## Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `search_messages` | Search for messages across chats | `query`, `chat_id`, `limit` |
| `get_recent_messages` | Get recent messages from a chat | `chat_id`, `limit` |
| `list_chats` | List all available conversations | `limit` |
| `send_message` | Send message to existing chat | `chat_id`, `message` |
| `send_message_to_number` | Send message to phone/email | `phone_number`, `message` |
| `get_contacts` | Retrieve contact list | `limit` |
| `mark_chat_read` | Mark a chat as read | `chat_id` |
| `get_server_info` | Get BlueBubbles server status | None |
| `get_chat_details` | Get detailed chat information | `chat_id` |

## Usage Examples

### Finding Recent Messages
```
Claude, show me the last 5 messages from my chat with John
```

### Sending a Message
```
Claude, send "Hello there!" to +1234567890
```

### Searching Messages
```
Claude, search for messages containing "meeting" from the last week
```

## Security Considerations

- Never commit passwords or sensitive URLs to version control
- Use environment variables or Docker secrets for configuration
- The server includes comprehensive error handling and input validation
- All API requests include proper timeout handling (30 seconds)

## Troubleshooting

### Common Issues

1. **Connection Failed**: Verify BlueBubbles URL and password are correct
2. **No Messages Found**: Check that BlueBubbles server has message access enabled
3. **Timeout Errors**: Ensure BlueBubbles server is responsive and network is stable
4. **404 Errors**: Currently, some of the endpoints are not correct. Feel free to make a PR!

### Logging

The server logs to stderr with INFO level by default. Check logs for detailed error information.

## Development

### File Structure
```
├── bluebubbles_server.py    # Main MCP server implementation
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container configuration
└── README.md               # This file
```

### Dependencies
- `mcp[cli]>=1.2.0` - Model Context Protocol framework
- `httpx` - Async HTTP client for API requests
- `python-dateutil` - Date parsing utilities

## License

This project is provided via the MIT license for integration with BlueBubbles and Claude Desktop.

## Contributing

Feel free to submit issues and pull requests to improve functionality and compatibility.