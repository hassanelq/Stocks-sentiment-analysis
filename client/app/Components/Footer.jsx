import React from "react";

const Footer = () => (
  <footer className="bg-gray-50">
    <div className="custom-screen py-10">
      <div className="border-t pt-6 flex flex-col sm:flex-row items-center justify-between text-center sm:text-left">
        {/* Footer Text */}
        <p className="text-gray-600 mb-4 sm:mb-0">
          Created by{" "}
          <a
            href="https://www.elqadi.me/"
            className="text-blue-600 hover:underline transition"
            target="_blank"
            rel="noopener noreferrer"
          >
            Hassan EL QADI
          </a>
          .
        </p>

        {/* Social Links */}
        <div className="flex items-center gap-4">
          <a
            href="https://www.linkedin.com/in/el-qadi/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-600 hover:text-blue-600 transition"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="w-6 h-6"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path d="M19 0H5C2.24 0 0 2.24 0 5v14c0 2.76 2.24 5 5 5h14c2.76 0 5-2.24 5-5V5c0-2.76-2.24-5-5-5zM8.5 19H5.5v-9h3v9zm-1.5-10.2c-.97 0-1.7-.8-1.7-1.7 0-.97.79-1.7 1.7-1.7s1.7.79 1.7 1.7c0 .96-.79 1.7-1.7 1.7zm13 10.2h-3v-4.7c0-1.12-.9-2-2-2s-2 .88-2 2v4.7h-3v-9h3v1.6c.61-.96 1.85-1.6 3-1.6 2.21 0 4 1.79 4 4v4.9z" />
            </svg>
          </a>
          <a
            href="https://github.com/hassanelq/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-600 hover:text-gray-800 transition"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="w-6 h-6"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path d="M12 2C6.48 2 2 6.48 2 12c0 4.42 2.87 8.17 6.84 9.5.5.09.66-.22.66-.48v-1.72c-2.78.6-3.37-1.34-3.37-1.34-.45-1.16-1.1-1.47-1.1-1.47-.9-.62.07-.61.07-.61 1 .07 1.54 1.03 1.54 1.03.89 1.53 2.34 1.09 2.91.83.09-.65.35-1.09.63-1.34-2.22-.26-4.56-1.11-4.56-4.95 0-1.09.39-1.98 1.02-2.68-.1-.25-.45-1.28.1-2.65 0 0 .84-.27 2.75 1.02A9.54 9.54 0 0 1 12 6.85a9.55 9.55 0 0 1 2.5.33c1.91-1.3 2.75-1.02 2.75-1.02.56 1.37.21 2.4.1 2.65.63.7 1.02 1.59 1.02 2.68 0 3.84-2.34 4.69-4.57 4.94.36.31.68.92.68 1.86v2.75c0 .27.18.58.68.48A10.02 10.02 0 0 0 22 12c0-5.52-4.48-10-10-10z" />
            </svg>
          </a>
          <a
            href="https://x.com/el_qadi_"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-600 hover:text-blue-400 transition"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="w-6 h-6"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path d="M22.46 6c-.77.35-1.6.59-2.46.69a4.28 4.28 0 0 0 1.88-2.37c-.8.48-1.69.82-2.64 1A4.22 4.22 0 0 0 15.5 4c-2.34 0-4.22 2.11-3.92 4.44A12.06 12.06 0 0 1 3.2 5a4.2 4.2 0 0 0-.57 2.13c0 1.47.73 2.77 1.85 3.53a4.2 4.2 0 0 1-1.91-.53v.05c0 2.07 1.48 3.8 3.44 4.19a4.2 4.2 0 0 1-1.9.07 4.22 4.22 0 0 0 3.95 2.94A8.45 8.45 0 0 1 2 19.29a12 12 0 0 0 6.29 1.85c7.55 0 11.67-6.25 11.67-11.67 0-.18 0-.36-.02-.54a8.3 8.3 0 0 0 2.02-2.11z" />
            </svg>
          </a>
          <a
            href="https://www.elqadi.me/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-600 hover:text-green-500 transition"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="w-6 h-6"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <path d="M12 2a10 10 0 0 0-10 10 10 10 0 1 0 10-10zm0 2c2.1 0 4 .81 5.47 2.13a12.29 12.29 0 0 0-10.94 0A8 8 0 0 1 12 4zM4 12c0-1.67.56-3.21 1.5-4.47a10.23 10.23 0 0 1 13 0A7.96 7.96 0 1 1 4 12zm7.5 6.33a2 2 0 1 1 4 0 2 2 0 0 1-4 0z" />
            </svg>
          </a>
        </div>
      </div>
    </div>
  </footer>
);

export default Footer;
