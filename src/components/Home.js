import React from 'react';
import './Home.css';
import About from './About';
import Features from './Features';

function Home() {
  return (
    <section className="home-container">
      <div className="home-content">
        <h1>CareerPath AI</h1>
        <p>Your Personalized Career Companion</p>
        <a href="#get-started" className="home-button">Get Started</a>
      </div>

      <About />
      <Features />
    </section>
  );
}

export default Home;
