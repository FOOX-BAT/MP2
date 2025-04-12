// src/components/Navbar.js
import React from 'react';
import './Navbar.css';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="logo">AI Career Path</div>
      <ul className="nav-links">
        <li><a href="#about">About</a></li>
        <li><a href="#profile">Profile</a></li>
        <li><a href="#chatbot">Chatbot</a></li>
      </ul>
    </nav>
  );
};

export default Navbar;
