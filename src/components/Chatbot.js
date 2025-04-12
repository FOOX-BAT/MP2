import React from 'react';
import './Chatbot.css';

const Chatbot = () => {
  const handleClick = () => {
    window.open('https://your-chatbot-link.com', '_blank');
  };

  return (
    <button className="chatbot-btn" onClick={handleClick}>
      💬
    </button>
  );
};

export default Chatbot;
