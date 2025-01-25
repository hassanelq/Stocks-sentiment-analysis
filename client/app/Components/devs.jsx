const Contributors = () => (
  <SectionWrapper>
    <div>
      <h2 className="text-3xl font-semibold text-gray-800 text-center">
        Meet the Contributors
      </h2>
      <div className="mt-4 flex flex-col items-center space-y-6">
        <div>
          <h3 className="text-xl font-bold text-gray-700">Hassan EL QADI</h3>
          <p className="text-gray-600">
            Lead Developer and NLP Specialist.
            <a
              href="https://www.elqadi.me/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline ml-2"
            >
              Learn more
            </a>
          </p>
        </div>
        <div>
          <h3 className="text-xl font-bold text-gray-700">Achraf HAJJI</h3>
          <p className="text-gray-600">
            Co-Developer and Data Engineer.
            <a
              href="mailto:achraf.hajji@edu.uiz.ac.ma"
              className="text-blue-600 hover:underline ml-2"
            >
              Contact
            </a>
          </p>
        </div>
      </div>
    </div>
  </SectionWrapper>
);

const SectionWrapper = ({ children, className = "" }) => (
  <section className={`py-16 ${className}`}>{children}</section>
);

export default Contributors;
