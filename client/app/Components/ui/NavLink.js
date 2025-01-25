import Link from "next/link";
import PropTypes from "prop-types";

const NavLink = ({ children, href, className = "", ...props }) => (
  <Link
    href={href}
    className={`py-2.5 px-4 text-center rounded-lg duration-150 ${className}`}
    {...props}
  >
    {children}
  </Link>
);

NavLink.propTypes = {
  children: PropTypes.node.isRequired,
  href: PropTypes.string.isRequired,
  className: PropTypes.string,
};

export default NavLink;
