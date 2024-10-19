import React, { useState, useEffect, useRef } from 'react';
import { Send } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [pdfFile, setPdfFile] = useState(null);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { text: input, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://0.0.0.0:3001/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: input }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      const botMessage = { text: data.response, sender: 'bot' };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = { text: 'Sorry, there was an error processing your request.', sender: 'bot' };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePdfUpload = async (e) => {
    e.preventDefault();
    if (!pdfFile) return;

    const formData = new FormData();
    formData.append('file', pdfFile);

    setIsLoading(true);

    try {
      const response = await fetch('http://0.0.0.0:3001/upload-pdf', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      const botMessage = { text: `PDF uploaded successfully: ${data.file_path}`, sender: 'bot' };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = { text: 'Sorry, there was an error processing your request.', sender: 'bot' };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setPdfFile(null);
    }
  };

  const handleFileUploadClick = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="flex flex-col h-screen w-4/5 mx-auto">
        <div className="flex justify-end p-4 border-b border-gray-200">
        <form onSubmit={handlePdfUpload} className="flex space-x-2">
          <button
            type="button"
            onClick={handleFileUploadClick}
            className="p-2 bg-green-500 text-white rounded-lg hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-green-500"
          >
            Select PDF
          </button>
          <input
            type="file"
            accept="application/pdf"
            ref={fileInputRef}
            onChange={(e) => setPdfFile(e.target.files[0])}
            className="hidden"
          />
          <button
            type="submit"
            disabled={isLoading}
            className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
          >
            Upload PDF
          </button>
        </form>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((message, index) => (
            <div key={index} className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`w-fit max-w-[75%] p-3 rounded-lg ${
                message.sender === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-800'
                }`}>
                {message.sender === 'user' ? (
                    <p className="text-sm sm:text-base">{message.text}</p>
                ) : (
                    <ReactMarkdown 
                    className="text-sm sm:text-base markdown-content" 
                    components={{
                        p: ({node, ...props}) => <p className="mb-2" {...props} />,
                        h1: ({node, ...props}) => <h1 className="text-2xl font-bold mb-2" {...props} />,
                        h2: ({node, ...props}) => <h2 className="text-xl font-bold mb-2" {...props} />,
                        h3: ({node, ...props}) => <h3 className="text-lg font-bold mb-2" {...props} />,
                        ul: ({node, ...props}) => <ul className="list-disc pl-4 mb-2" {...props} />,
                        ol: ({node, ...props}) => <ol className="list-decimal pl-4 mb-2" {...props} />,
                        li: ({node, ...props}) => <li className="mb-1" {...props} />,
                        code: ({node, inline, ...props}) => (
                        inline 
                            ? <code className="bg-gray-100 text-red-500 px-1 rounded" {...props} />
                            : <code className="block bg-gray-100 p-2 rounded mb-2 overflow-x-auto" {...props} />
                        )
                    }}
                    >
                    {message.text}
                    </ReactMarkdown>
                )}
                </div>
            </div>
            ))}
            {isLoading && (
            <div className="flex justify-start">
                <div className="w-fit max-w-[75%] p-3 rounded-lg bg-gray-200 text-gray-800">
                <p className="text-sm sm:text-base">Thinking...</p>
                </div>
            </div>
            )}
            <div ref={messagesEndRef} />
        </div>
        <form onSubmit={handleSubmit} className="p-4 border-t border-gray-200">
            <div className="flex space-x-2">
            <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your message..."
                className="flex-1 p-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
                type="submit"
                disabled={isLoading}
                className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            >
                <Send size={24} />
            </button>
            </div>
        </form>
    </div>
  );
};

export default ChatInterface;