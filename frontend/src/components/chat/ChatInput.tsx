/**
 * ChatInput Component
 *
 * Provides a text input field and a send button for users to interact with the chatbot.
 */

import React from "react";
import { Send } from "lucide-react";

interface ChatInputProps {
  message: string;
  setMessage: (message: string) => void;
  onSendMessage: () => void;
  isPending: boolean;
  currentSymbol: string | null;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  message,
  setMessage,
  onSendMessage,
  isPending,
  currentSymbol,
}) => {
  return (
    <div className="border-t border-gray-200 px-6 py-4 bg-white">
      <div className="flex gap-3">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onSendMessage()}
          placeholder={
            currentSymbol
              ? `Ask about ${currentSymbol} or request analysis...`
              : "Ask questions or search for a symbol on the right..."
          }
          className="flex-1 border border-gray-300 rounded-xl px-4 py-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-shadow"
          disabled={isPending}
        />
        <button
          onClick={onSendMessage}
          disabled={!message.trim() || isPending}
          className="bg-gradient-to-r from-blue-500 to-blue-600 text-white px-5 py-3 rounded-xl hover:from-blue-600 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-sm hover:shadow-md"
        >
          <Send className="h-5 w-5" />
        </button>
      </div>
    </div>
  );
};
