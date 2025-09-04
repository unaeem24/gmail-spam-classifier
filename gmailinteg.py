from __future__ import print_function
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
import email
from email import policy
from bs4 import BeautifulSoup
import re

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

class GmailIntegration:
    def __init__(self, token_file='token.json', credentials_file='credentials.json'):
        self.token_file = token_file
        self.credentials_file = credentials_file
        self.service = None
        self.creds = None
        
    def authenticate(self):
        """Authenticate with Gmail API"""
        # The file token.json stores the user's access and refresh tokens
        if os.path.exists(self.token_file):
            self.creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
        
        # If there are no (valid) credentials available, let the user log in.
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES)
                self.creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(self.token_file, 'w') as token:
                token.write(self.creds.to_json())
        
        try:
            # Call the Gmail API
            self.service = build('gmail', 'v1', credentials=self.creds)
            print("Gmail authentication successful")
            return True
        except HttpError as error:
            print(f'An error occurred during authentication: {error}')
            return False
    
    def get_email_content(self, message_id):
        """Get email content by message ID"""
        try:
            message = self.service.users().messages().get(
                userId='me', id=message_id, format='raw').execute()
            
            # Decode the raw email content
            msg_str = base64.urlsafe_b64decode(message['raw'].encode('ASCII'))
            mime_msg = email.message_from_bytes(msg_str, policy=policy.default)
            
            return self._parse_email(mime_msg)
        except HttpError as error:
            print(f'An error occurred: {error}')
            return None
    
    def _parse_email(self, mime_msg):
        """Parse MIME message to extract relevant content"""
        email_data = {
            'id': None,
            'subject': '',
            'from': '',
            'date': '',
            'body': '',
            'snippet': '',
            'labels': []
        }
        
        # Extract headers
        email_data['subject'] = mime_msg.get('subject', '')
        email_data['from'] = mime_msg.get('from', '')
        email_data['date'] = mime_msg.get('date', '')
        
        # Extract body content
        email_data['body'] = self._extract_body(mime_msg)
        email_data['snippet'] = email_data['body'][:200] + '...' if len(email_data['body']) > 200 else email_data['body']
        
        return email_data
    
    def _extract_body(self, mime_msg):
        """Extract text body from MIME message"""
        body = ""
        
        if mime_msg.is_multipart():
            for part in mime_msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                # Skip attachments
                if "attachment" not in content_disposition:
                    # Prefer text/plain over text/html
                    if content_type == "text/plain":
                        body = part.get_payload(decode=True).decode()
                        break
                    elif content_type == "text/html":
                        body = part.get_payload(decode=True).decode()
                        # Convert HTML to plain text
                        body = self._html_to_text(body)
        else:
            # Not multipart
            body = mime_msg.get_payload(decode=True).decode()
            if mime_msg.get_content_type() == "text/html":
                body = self._html_to_text(body)
        
        return body
    
    def _html_to_text(self, html_content):
        """Convert HTML content to plain text"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading/trailing space
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def list_messages(self, max_results=10, query=None):
        """List messages from Gmail - FIXED VERSION"""
        try:
            # Build the request parameters
            params = {
                'userId': 'me',
                'maxResults': max_results
            }
            
            # Add query if provided
            if query:
                params['q'] = query
                
            results = self.service.users().messages().list(**params).execute()
            
            messages = results.get('messages', [])
            return messages
        except HttpError as error:
            print(f'An error occurred: {error}')
            return []
    
    def get_recent_emails(self, max_results=10):
        """Get recent emails from inbox"""
        messages = self.list_messages(max_results=max_results)
        emails = []
        
        for message in messages:
            email_data = self.get_email_content(message['id'])
            if email_data:
                email_data['id'] = message['id']
                emails.append(email_data)
        
        return emails
    
    def get_emails_by_label(self, label_name, max_results=10):
        """Get emails by label name - FIXED VERSION"""
        # First, get label ID from name
        try:
            labels = self.service.users().labels().list(userId='me').execute().get('labels', [])
            label_id = None
            
            for label in labels:
                if label['name'].lower() == label_name.lower():
                    label_id = label['id']
                    break
            
            if label_id:
                # Use query to filter by label
                query = f"label:{label_name}"
                messages = self.list_messages(max_results=max_results, query=query)
                emails = []
                
                for message in messages:
                    email_data = self.get_email_content(message['id'])
                    if email_data:
                        email_data['id'] = message['id']
                        emails.append(email_data)
                
                return emails
            else:
                print(f"Label '{label_name}' not found")
                return []
        except HttpError as error:
            print(f"Error getting label '{label_name}': {error}")
            return []

# Example usage
if __name__ == "__main__":
    gmail = GmailIntegration()
    if gmail.authenticate():
        # Get recent emails
        emails = gmail.get_recent_emails(max_results=5)
        
        for i, email_data in enumerate(emails):
            print(f"Email {i+1}:")
            print(f"From: {email_data['from']}")
            print(f"Subject: {email_data['subject']}")
            print(f"Snippet: {email_data['snippet']}")
            print("---")