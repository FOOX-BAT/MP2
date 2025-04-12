import React from 'react';
import './ProfileButton.css';

const ProfileButton = () => {
  const handleClick = () => {
    // Replace with your profile page URL or keep as placeholder
    window.open('https://your-profile-link.com', '_blank');
  };

  return (
    <button className="profile-btn" onClick={handleClick}>
      Profile
    </button>
  );
};

export default ProfileButton;
