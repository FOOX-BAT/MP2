import React from 'react';
import './Contact.css';

function Contact() {
  return (
    <section className="contact-section" id="contact">
      <h2>Contact Us</h2>
      <p>Have questions or suggestions? We'd love to hear from you!</p>
      <form className="contact-form">
        <input type="text" placeholder="Your Name" required />
        <input type="email" placeholder="Email Address" required />
        <textarea placeholder="Your Message" required></textarea>
        <button type="submit">Send Message</button>
      </form>
    </section>
  );
}

export default Contact;
