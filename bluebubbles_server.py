#!/usr/bin/env python3
"""
Simple BlueBubbles MCP Server - Interface for BlueBubbles iMessage API
"""
import os
import sys
import logging
from datetime import datetime, timezone
import httpx
import json
from mcp.server.fastmcp import FastMCP
from dateutil import parser
import uuid

# Configure logging to stderr
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	stream=sys.stderr
)
logger = logging.getLogger("bluebubbles-server")

# Initialize MCP server - NO PROMPT PARAMETER!
mcp = FastMCP("bluebubbles")

# Configuration
BLUEBUBBLES_URL = os.environ.get("BLUEBUBBLES_URL", "")
BLUEBUBBLES_PASSWORD = os.environ.get("BLUEBUBBLES_PASSWORD", "")

# === UTILITY FUNCTIONS ===

def get_base_url():
	"""Get the properly formatted base URL for BlueBubbles API."""
	if not BLUEBUBBLES_URL:
		return ""
	url = BLUEBUBBLES_URL.rstrip('/')
	if not url.startswith('http'):
		url = f"http://{url}"
	return url

async def make_api_request(endpoint: str, method: str = "GET", data: dict = None):
	"""Make a request to the BlueBubbles API."""
	base_url = get_base_url()
	if not base_url:
		raise ValueError("BlueBubbles URL not configured")
	
	url = f"{base_url}/api/v1/{endpoint.lstrip('/')}?password={BLUEBUBBLES_PASSWORD}"
	headers = {}
	
	async with httpx.AsyncClient() as client:
		if method == "GET":
			response = await client.get(url, headers=headers, timeout=30)
		elif method == "POST":
			response = await client.post(url, headers=headers, json=data, timeout=30)
		else:
			raise ValueError(f"Unsupported HTTP method: {method}")
		
		response.raise_for_status()
		return response.json()

def format_message(msg):
	"""Format a message object for display."""
	text = msg.get('text', '').strip()
	if not text:
		text = "[No text content]"
	sender = msg.get('handle', {}).get('address', 'Unknown')
	date = msg.get('dateCreated', '')
	is_from_me = msg.get('isFromMe', False)
	
	if is_from_me:
		sender = "Me"
	
	try:
		dt = parser.parse(date)
		date_str = dt.strftime('%Y-%m-%d %H:%M')
	except:
		date_str = date
	
	return f"[{date_str}] {sender}: {text}"

def format_chat(chat):
	"""Format a chat object for display."""
	display_name = chat.get('displayName', '')
	chat_identifier = chat.get('chatIdentifier', '')
	participant_count = len(chat.get('participants', []))
	
	if display_name:
		return f"{display_name} ({participant_count} participants)"
	elif chat_identifier:
		return f"{chat_identifier} ({participant_count} participants)"
	else:
		return f"Chat ({participant_count} participants)"

# === MCP TOOLS ===

@mcp.tool()
async def search_messages(query: str = "", chat_id: str = "", limit: str = "20") -> str:
	"""Search for messages containing specific text across all chats or in a specific chat."""
	logger.info(f"Searching messages with query: {query}, chat_id: {chat_id}, limit: {limit}")
	
	if not query.strip():
		return "❌ Error: Query text is required"
	
	try:
		limit_int = int(limit) if limit.strip() else 20
		limit_int = min(limit_int, 1000)
		
		params = {
			"limit": limit_int,
			"where": {
				"statement": "message.text = :text",
				"args": {
					"text": query
				}
			}
		}
		
		if chat_id.strip():
			params["chatGuid"] = chat_id
		
		data = await make_api_request("message/query", "POST", params)
		messages = data.get('data', [])
		
		if not messages:
			return f"🔍 No messages found containing '{query}'"
		
		result = f"🔍 Found {len(messages)} message(s) containing '{query}':\n\n"
		for msg in messages[:limit_int]:
			result += format_message(msg) + "\n"
		
		return result
		
	except Exception as e:
		logger.error(f"Error searching messages: {e}")
		return f"❌ Error searching messages: {str(e)}"

@mcp.tool()
async def get_recent_messages(chat_id: str = "", limit: str = "10") -> str:
	"""Get recent messages from all chats or a specific chat."""
	logger.info(f"Getting recent messages for chat_id: {chat_id}, limit: {limit}")
	
	try:
		limit_int = int(limit) if limit.strip() else 10
		limit_int = min(limit_int, 50)
		
		if chat_id.strip():
			# Get messages from specific chat
			data = await make_api_request(f"chat/{chat_id}/message")
		else:
			# Get messages from all chats
			data = await make_api_request(f"message/query")
		
		messages = data.get('data', [])
		
		if not messages:
			return "📊 No recent messages found"
		
		result = f"📊 Recent messages ({len(messages)} shown):\n\n"
		for msg in messages:
			result += format_message(msg) + "\n"
		
		return result
		
	except Exception as e:
		logger.error(f"Error getting recent messages: {e}")
		return f"❌ Error getting recent messages: {str(e)}"


#not sure this is anything
@mcp.tool()
async def list_chats(limit: str = "20") -> str:
	"""List all available chats/conversations."""
	logger.info(f"Listing chats with limit: {limit}")
	
	try:
		limit_int = int(limit) if limit.strip() else 20
		limit_int = min(limit_int, 100)
		
		data = await make_api_request(f"chat/query", "POST")
		chats = data.get('data', [])
		
		if not chats:
			return "📁 No chats found"
		
		result = f"📁 Available chats ({len(chats)} shown):\n\n"
		for i, chat in enumerate(chats, 1):
			chat_id = chat.get('guid', '')
			formatted = format_chat(chat)
			result += f"{i}. {formatted}\n   ID: {chat_id}\n\n"
		
		return result
		
	except Exception as e:
		logger.error(f"Error listing chats: {e}")
		return f"❌ Error listing chats: {str(e)}"

@mcp.tool()
async def send_message(chat_id: str = "", message: str = "") -> str:
	"""Send a message to a specific chat."""
	logger.info(f"Sending message to chat_id: {chat_id}")
	
	if not chat_id.strip():
		return "❌ Error: Chat ID is required"
	
	if not message.strip():
		return "❌ Error: Message text is required"
	
	try:
		data = {
			"chatGuid": chat_id,
			"tempGuid": str(uuid.uuid4()),
			"message": message
		}
		
		response = await make_api_request("message/text", "POST", data)
		
		if response.get('status') == 200:
			return f"✅ Message sent successfully to {chat_id}"
		else:
			return f"❌ Failed to send message: {response.get('message', 'Unknown error')}"
		
	except Exception as e:
		logger.error(f"Error sending message: {e}")
		return f"❌ Error sending message: {str(e)}"

@mcp.tool()
async def send_message_to_number(phone_number: str = "", message: str = "") -> str:
	"""Send a message directly to a phone number or email address."""
	logger.info(f"Sending message to number: {phone_number}")
	
	if not phone_number.strip():
		return "❌ Error: Phone number or email is required"
	
	if not message.strip():
		return "❌ Error: Message text is required"
	
	try:
		data = {
			"addresses": [phone_number],
			"message": message
		}
		
		response = await make_api_request("message/text/new", "POST", data)
		
		if response.get('status') == 200:
			return f"✅ Message sent successfully to {phone_number}"
		else:
			return f"❌ Failed to send message: {response.get('message', 'Unknown error')}"
		
	except Exception as e:
		logger.error(f"Error sending message: {e}")
		return f"❌ Error sending message: {str(e)}"

@mcp.tool()
async def get_contacts(limit: str = "50") -> str:
	"""Get the list of contacts from BlueBubbles."""
	logger.info(f"Getting contacts with limit: {limit}")
	
	try:
		limit_int = int(limit) if limit.strip() else 50
		limit_int = min(limit_int, 200)
		
		data = await make_api_request(f"contact")
		contacts = data.get('data', [])
		
		if not contacts:
			return "📊 No contacts found"
		
		result = f"📊 Contacts ({len(contacts)} shown):\n\n"
		for contact in contacts:
			first_name = contact.get('firstName', '')
			last_name = contact.get('lastName', '')
			name = f"{first_name} {last_name}".strip()
			if not name:
				name = "Unknown"
			
			phones = contact.get('phoneNumbers', [])
			emails = contact.get('emails', [])
			
			result += f"• {name}\n"
			for phone in phones:
				result += f"  📱 {phone.get('address', '')}\n"
			for email in emails:
				result += f"  📧 {email.get('address', '')}\n"
			result += "\n"
		
		return result
		
	except Exception as e:
		logger.error(f"Error getting contacts: {e}")
		return f"❌ Error getting contacts: {str(e)}"

@mcp.tool()
async def mark_chat_read(chat_id: str = "") -> str:
	"""Mark all messages in a chat as read."""
	logger.info(f"Marking chat as read: {chat_id}")
	
	if not chat_id.strip():
		return "❌ Error: Chat ID is required"
	
	try:
		response = await make_api_request(f"chat/{chat_id}/read", "POST", {})
		
		if response.get('status') == 200:
			return f"✅ Chat {chat_id} marked as read"
		else:
			return f"❌ Failed to mark chat as read: {response.get('message', 'Unknown error')}"
		
	except Exception as e:
		logger.error(f"Error marking chat as read: {e}")
		return f"❌ Error marking chat as read: {str(e)}"

@mcp.tool()
async def get_server_info() -> str:
	"""Get information about the BlueBubbles server."""
	logger.info("Getting server info")
	
	try:
		data = await make_api_request("server/info")
		info = data.get('data', {})
		
		result = "🌐 BlueBubbles Server Information:\n\n"
		result += f"• OS Version: {info.get('os_version', 'Unknown')}\n"
		result += f"• Server Version: {info.get('server_version', 'Unknown')}\n"
		result += f"• Private API: {'Enabled' if info.get('private_api', False) else 'Disabled'}\n"
		result += f"• Proxy Service: {info.get('proxy_service', 'Unknown')}\n"
		
		return result
		
	except Exception as e:
		logger.error(f"Error getting server info: {e}")
		return f"❌ Error getting server info: {str(e)}"

@mcp.tool()
async def get_chat_details(chat_id: str = "") -> str:
	"""Get detailed information about a specific chat."""
	logger.info(f"Getting details for chat: {chat_id}")
	
	if not chat_id.strip():
		return "❌ Error: Chat ID is required"
	
	try:
		data = await make_api_request(f"chat/{chat_id}")
		chat = data.get('data', {})
		
		if not chat:
			return f"❌ Chat {chat_id} not found"
		
		result = f"📁 Chat Details:\n\n"
		result += f"• Display Name: {chat.get('displayName', 'N/A')}\n"
		result += f"• Chat ID: {chat.get('guid', '')}\n"
		result += f"• Is Group: {'Yes' if chat.get('isGroup', False) else 'No'}\n"
		
		participants = chat.get('participants', [])
		if participants:
			result += f"\n👥 Participants ({len(participants)}):\n"
			for p in participants:
				address = p.get('address', 'Unknown')
				result += f"  • {address}\n"
		
		return result
		
	except Exception as e:
		logger.error(f"Error getting chat details: {e}")
		return f"❌ Error getting chat details: {str(e)}"

# === SERVER STARTUP ===
if __name__ == "__main__":
	logger.info("Starting BlueBubbles MCP server...")
	
	if not BLUEBUBBLES_URL:
		logger.warning("BLUEBUBBLES_URL not set - server will need configuration")
	
	if not BLUEBUBBLES_PASSWORD:
		logger.warning("BLUEBUBBLES_PASSWORD not set - authentication may fail")
	
	try:
		mcp.run(transport='stdio')
	except Exception as e:
		logger.error(f"Server error: {e}", exc_info=True)
		sys.exit(1)