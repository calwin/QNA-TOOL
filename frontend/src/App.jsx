import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, FileText, Loader2, Bot, User, FileQuestion } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './App.css';

function App() {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [chatHistory, setChatHistory] = useState([
    { type: 'bot', content: 'Hello! Upload your documents (PDF, Excel) and ask me anything about them.' }
  ]);
  const [question, setQuestion] = useState('');
  const [asking, setAsking] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory]);

  const handleFileChange = async (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFiles = Array.from(e.target.files);
      setUploading(true);
      
      const formData = new FormData();
      selectedFiles.forEach(file => {
        formData.append('files', file);
      });

      try {
        const response = await fetch('http://localhost:8000/upload', {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) throw new Error('Upload failed');
        
        const data = await response.json();
        setFiles(prev => [...prev, ...selectedFiles]);
        setChatHistory(prev => [...prev, { type: 'bot', content: `Successfully processed ${selectedFiles.length} files.` }]);
      } catch (error) {
        console.error(error);
        setChatHistory(prev => [...prev, { type: 'bot', content: 'Error uploading files. Please try again.' }]);
      } finally {
        setUploading(false);
      }
    }
  };

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!question.trim() || asking) return;

    const currentQuestion = question;
    setQuestion('');
    setChatHistory(prev => [...prev, { type: 'user', content: currentQuestion }]);
    setAsking(true);

    try {
      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: currentQuestion }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get answer');
      }

      const data = await response.json();
      setChatHistory(prev => [...prev, { 
        type: 'bot', 
        content: data.answer, 
        sources: data.sources 
      }]);
    } catch (error) {
      console.error(error);
      setChatHistory(prev => [...prev, { type: 'bot', content: `Error: ${error.message}` }]);
    } finally {
      setAsking(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header glass-panel">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <FileQuestion size={24} color="#60a5fa" />
          <h1>Q&A Intelligence</h1>
        </div>
        <div style={{ fontSize: '0.9rem', opacity: 0.7 }}>
          Azure OpenAI Powered
        </div>
      </header>

      <div className="main-content">
        {/* Sidebar / Upload */}
        <div className="sidebar glass-panel" style={{ padding: '1rem' }}>
          <h3>Documents</h3>
          <div 
            className={`upload-area ${uploading ? 'dragging' : ''}`}
            onClick={() => fileInputRef.current.click()}
          >
            <input
              type="file"
              multiple
              ref={fileInputRef}
              style={{ display: 'none' }}
              onChange={handleFileChange}
              accept=".pdf,.xlsx,.xls,.xml,.dbc,.arxml,.cdd,.a2l,.py"
            />
            {uploading ? (
              <Loader2 className="animate-spin" size={32} style={{ margin: '0 auto' }} />
            ) : (
              <Upload size={32} style={{ margin: '0 auto', opacity: 0.7 }} />
            )}
            <p style={{ marginTop: '0.5rem', fontSize: '0.9rem' }}>
              {uploading ? 'Processing...' : 'Click to upload files'}
            </p>
            <p style={{ fontSize: '0.75rem', opacity: 0.7, marginTop: '0.25rem' }}>
              PDF, Excel, XML, DBC, ARXML, CDD, A2L, Python
            </p>
          </div>

          <div className="file-list">
            {files.map((file, index) => (
              <div key={index} className="file-item">
                <FileText size={16} />
                <span style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                  {file.name}
                </span>
              </div>
            ))}
            {files.length === 0 && (
              <div style={{ textAlign: 'center', opacity: 0.5, marginTop: '2rem' }}>
                No documents uploaded
              </div>
            )}
          </div>
        </div>

        {/* Chat Area */}
        <div className="chat-container glass-panel">
          <div className="messages-area">
            {chatHistory.map((msg, index) => (
              <div key={index} className={`message ${msg.type}`}>
                <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.25rem', alignItems: 'center', fontSize: '0.8rem', opacity: 0.8 }}>
                  {msg.type === 'bot' ? <Bot size={14} /> : <User size={14} />}
                  <span>{msg.type === 'bot' ? 'AI Assistant' : 'You'}</span>
                </div>
                <div className="message-content">
                  {msg.type === 'bot' ? (
                    <ReactMarkdown
                      components={{
                        code({ node, inline, className, children, ...props }) {
                          const match = /language-(\w+)/.exec(className || '');
                          return !inline && match ? (
                            <SyntaxHighlighter
                              style={vscDarkPlus}
                              language={match[1]}
                              PreTag="div"
                              customStyle={{
                                borderRadius: '8px',
                                padding: '1rem',
                                margin: '0.5rem 0',
                                fontSize: '0.9rem'
                              }}
                              {...props}
                            >
                              {String(children).replace(/\n$/, '')}
                            </SyntaxHighlighter>
                          ) : (
                            <code className={className} style={{
                              background: 'rgba(255, 255, 255, 0.1)',
                              padding: '0.2rem 0.4rem',
                              borderRadius: '4px',
                              fontSize: '0.9em',
                              fontFamily: 'monospace'
                            }} {...props}>
                              {children}
                            </code>
                          );
                        },
                        p({ children }) {
                          return <p style={{ margin: '0.5rem 0', lineHeight: '1.6' }}>{children}</p>;
                        },
                        ul({ children }) {
                          return <ul style={{ margin: '0.5rem 0', paddingLeft: '1.5rem' }}>{children}</ul>;
                        },
                        ol({ children }) {
                          return <ol style={{ margin: '0.5rem 0', paddingLeft: '1.5rem' }}>{children}</ol>;
                        },
                        li({ children }) {
                          return <li style={{ margin: '0.25rem 0' }}>{children}</li>;
                        },
                        h1({ children }) {
                          return <h1 style={{ fontSize: '1.5rem', fontWeight: 'bold', margin: '1rem 0 0.5rem' }}>{children}</h1>;
                        },
                        h2({ children }) {
                          return <h2 style={{ fontSize: '1.3rem', fontWeight: 'bold', margin: '0.8rem 0 0.4rem' }}>{children}</h2>;
                        },
                        h3({ children }) {
                          return <h3 style={{ fontSize: '1.1rem', fontWeight: 'bold', margin: '0.6rem 0 0.3rem' }}>{children}</h3>;
                        },
                        strong({ children }) {
                          return <strong style={{ fontWeight: 'bold', color: '#60a5fa' }}>{children}</strong>;
                        },
                        blockquote({ children }) {
                          return <blockquote style={{
                            borderLeft: '3px solid #60a5fa',
                            paddingLeft: '1rem',
                            margin: '0.5rem 0',
                            fontStyle: 'italic',
                            opacity: 0.9
                          }}>{children}</blockquote>;
                        }
                      }}
                    >
                      {msg.content}
                    </ReactMarkdown>
                  ) : (
                    <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                  )}
                </div>
                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources">
                    <strong>Sources:</strong>
                    <ul style={{ margin: '0.25rem 0 0 1rem', padding: 0 }}>
                      {msg.sources.map((source, idx) => (
                        <li key={idx}>{source}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ))}
            {asking && (
              <div className="message bot">
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  <Loader2 className="animate-spin" size={16} />
                  <span>Thinking...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form className="input-area" onSubmit={handleAsk}>
            <input
              type="text"
              placeholder="Ask a question about your documents..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={asking}
            />
            <button 
              type="submit" 
              className="btn btn-primary" 
              disabled={!question.trim() || asking}
            >
              <Send size={18} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
