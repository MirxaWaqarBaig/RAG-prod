#!/bin/bash

# Text prompt to send
PROMPT="我要退车，如何退"
DETAILED="false"  # true or "false"

# Send a POST request to the Graph RAG gateway

#localhost:8000/api/text-to-text
#http://chatbot.sharestyleai.com:8000/api/text-to-text
curl -X POST http://localhost:8000/api/text-to-text \
     -H "Content-Type: application/json" \
     -d "{\"prompt\": \"$PROMPT\", \"detailed\": $DETAILED}"
