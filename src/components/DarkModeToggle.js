import React from 'react';

function DarkModeToggle({ darkMode, setDarkMode }) {
  const handleChange = () => {
    setDarkMode(!darkMode);
  };

  return (
    <div>
      <label>
        Dark Mode
        <input
          type="checkbox"
          checked={darkMode}
          onChange={handleChange}
        />
      </label>
    </div>
  );
}

export default DarkModeToggle;
