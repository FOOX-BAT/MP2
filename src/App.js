import React from 'react';
import './App.css';
import Home from './components/Home';
import GetStarted from './components/GetStarted';
import Contact from './components/Contact';
import Footer from './components/Footer';
import Chatbot from './components/Chatbot';
import ProfileButton from './components/ProfileButton'; // ✅ Import

function App() {
  return (
    <div className="App">
      <ProfileButton /> {/* ✅ Add fixed top-right Profile button */}
      <Home />
      <GetStarted />
      <Contact />
      <Footer />
      <Chatbot />
    </div>
  );
}

export default App;
