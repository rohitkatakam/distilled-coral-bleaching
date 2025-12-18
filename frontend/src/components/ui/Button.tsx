import React from 'react';
import { motion } from 'framer-motion';
import type { HTMLMotionProps } from 'framer-motion';

interface ButtonProps extends HTMLMotionProps<"button"> {
  variant?: 'primary' | 'secondary' | 'outline';
  children: React.ReactNode;
  href?: string;
  target?: string;
  rel?: string;
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  children,
  className = '',
  href,
  target,
  rel,
  ...props
}) => {
  const baseStyles = "inline-flex items-center px-6 py-3 rounded-md font-semibold transition-colors duration-200 cursor-pointer";

  const variants = {
    primary: "bg-black text-white hover:bg-gray-800",
    secondary: "bg-gray-700 text-white hover:bg-gray-800",
    outline: "border-2 border-black text-black hover:bg-gray-100"
  };

  const Component = href ? motion.a : motion.button;

  return (
    // @ts-ignore - motion component types are tricky with dynamic "as" behavior equivalent
    <Component
      href={href}
      target={target}
      rel={rel}
      className={`${baseStyles} ${variants[variant]} ${className}`}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      {...(props as any)}
    >
      {children}
    </Component>
  );
};
