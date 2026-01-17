import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
import requests
from access_token_store import save_access_token

# CONFIG
api_key = 'e6e8a630-f1b0-4c7b-9c13-a0ef3684632d'
secret_key = 'n7wzc88d2h'
redirect_uri = 'http://localhost:8787/callback'
port = 8787

def get_auth_code():
    encoded_redirect_uri = urllib.parse.quote(redirect_uri, safe='')
    auth_url = f'https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={api_key}&redirect_uri={encoded_redirect_uri}'
    print(f"Opening browser:\n{auth_url}\n")
    webbrowser.open(auth_url)

    class OAuthCallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed_url = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(parsed_url.query)
            auth_code = query.get('code', [None])[0]

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"<h2>Authorization complete. You may now close this window.</h2>")

            print(f"✅ Received Auth Code: {auth_code}")
            self.server.auth_code = auth_code

    print(f"🚀 Waiting for callback on {redirect_uri} ...")
    httpd = HTTPServer(('localhost', port), OAuthCallbackHandler)
    httpd.handle_request()
    return httpd.auth_code

def exchange_token(auth_code):
    print("🔁 Exchanging code for token...")
    token_url = 'https://api.upstox.com/v2/login/authorization/token'
    data = {
        'code': auth_code,
        'client_id': api_key,
        'client_secret': secret_key,
        'redirect_uri': redirect_uri,
        'grant_type': 'authorization_code'
    }

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    response = requests.post(token_url, data=data, headers=headers)
    if response.status_code == 200:
        token_data = response.json()
        token = token_data.get('access_token')
        print(f"✅ Access Token: {token}")
        save_access_token(token)
        return token
    else:
        print("❌ Token exchange failed", response.status_code, response.text)
        return None

if __name__ == '__main__':
    auth_code = get_auth_code()
    if auth_code:
        exchange_token(auth_code)
