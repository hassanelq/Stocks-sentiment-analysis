import { FaLinkedin, FaGithub, FaGlobe } from "react-icons/fa";

const Contributors = () => (
  <SectionWrapper>
    <div>
      <h2 className="text-gray-800 text-3xl font-semibold sm:text-4xl text-center mb-8 ">
        Meet the Contributors
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-8 mx-4 md:mx-32">
        {/* Contributor 1 */}
        <ContributorCard
          name="Hassan EL QADI"
          role="Developer"
          photo="/images/Hassan.jpeg"
          socials={{
            linkedin: "https://www.linkedin.com/in/el-qadi/",
            github: "https://github.com/hassanelq",
            website: "https://www.elqadi.me/",
          }}
        />
        {/* Contributor 2 */}
        <ContributorCard
          name="Achraf HAJJI"
          role="Developer"
          photo="/images/Achraf.jpeg"
          socials={{
            linkedin: "https://www.linkedin.com/in/achraf-hajji/",
            github: "https://github.com/hajji-achraf/",
          }}
        />
      </div>
    </div>
  </SectionWrapper>
);

const ContributorCard = ({ name, role, photo, socials }) => (
  <div className="flex flex-col items-center bg-white shadow-lg rounded-lg p-6">
    <img
      src={photo}
      alt={`${name}'s photo`}
      className="w-24 h-24 rounded-full mb-4 object-cover shadow-md"
    />
    <h3 className="text-xl font-bold text-gray-800">{name}</h3>
    <p className="text-gray-600 text-center mb-4">{role}</p>
    <div className="flex space-x-4">
      {socials.linkedin && (
        <a
          href={socials.linkedin}
          target="_blank"
          rel="noopener noreferrer"
          className="text-gray-700 hover:text-gray-900"
        >
          <FaLinkedin size={32} />
        </a>
      )}
      {socials.github && (
        <a
          href={socials.github}
          target="_blank"
          rel="noopener noreferrer"
          className="text-gray-700 hover:text-gray-900"
        >
          <FaGithub size={32} />
        </a>
      )}
      {socials.website && (
        <a
          href={socials.website}
          target="_blank"
          rel="noopener noreferrer"
          className="text-gray-700 hover:text-gray-900"
        >
          <FaGlobe size={32} />
        </a>
      )}
    </div>
  </div>
);

const SectionWrapper = ({ children, className = "" }) => (
  <section className={`pt-8 pb-16 ${className}`}>{children}</section>
);

export default Contributors;
