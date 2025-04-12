import React from 'react';
import './Features.css';

function Features() {
  return (
    <section className="features-section" id="features">
      <h2>Features</h2>
      <div className="features-container">
        <div className="feature">
          <h3>Personalized Guidance</h3>
          <p>Receive tailored career advice based on your interests and skills.</p>
        </div>
        <div className="feature">
          <h3>Interactive Chatbot</h3>
          <p>Chat with our AI to explore various career paths in detail.</p>
        </div>
        <div className="feature">
          <h3>Profile Management</h3>
          <p>Manage and update your career profile to refine your suggestions.</p>
        </div>
      </div>
    </section>
  );
}

export default Features;
