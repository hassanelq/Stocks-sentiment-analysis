import React from "react";
import PropTypes from "prop-types";
import clsx from "clsx";

const Button = React.forwardRef(
  (
    {
      variant = "default",
      size = "default",
      className,
      children,
      disabled = false,
      ...props
    },
    ref
  ) => {
    const baseStyles =
      "inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50";

    const variantStyles = {
      default: "bg-blue-500 text-white hover:bg-blue-400",
      destructive: "bg-red-500 text-white hover:bg-red-400",
      outline: "border border-gray-300 bg-white hover:bg-gray-100",
      secondary: "bg-gray-500 text-white hover:bg-gray-400",
      ghost: "bg-transparent hover:bg-gray-100",
      link: "text-blue-500 underline hover:text-blue-400",
    };

    const sizeStyles = {
      default: "h-10 px-4",
      sm: "h-8 px-3",
      lg: "h-12 px-6",
      icon: "h-10 w-10",
    };

    const classes = clsx(
      baseStyles,
      variantStyles[variant],
      sizeStyles[size],
      className
    );

    return (
      <button className={classes} disabled={disabled} ref={ref} {...props}>
        {children}
      </button>
    );
  }
);

Button.displayName = "Button";

Button.propTypes = {
  variant: PropTypes.oneOf([
    "default",
    "destructive",
    "outline",
    "secondary",
    "ghost",
    "link",
  ]),
  size: PropTypes.oneOf(["default", "sm", "lg", "icon"]),
  className: PropTypes.string,
  disabled: PropTypes.bool,
  children: PropTypes.node.isRequired,
};

export default Button;
