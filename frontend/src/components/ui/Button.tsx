import React from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';

interface ButtonProps extends HTMLMotionProps<"button"> {
  variant?: 'primary' | 'secondary' | 'outline';
  children: React.ReactNode;
  href?: string;
}

export const Button: React.FC<ButtonProps> = ({ 
  variant = 'primary', 
  children, 
  className = '', 
  href,
  ...props 
}) => {
  const baseStyles = "inline-flex items-center px-6 py-3 rounded-full font-semibold transition-colors duration-200 cursor-pointer";
  
  const variants = {
    primary: "bg-coral-500 text-white hover:bg-coral-600 shadow-lg shadow-coral-500/30",
    secondary: "bg-ocean-700 text-white hover:bg-ocean-800 shadow-lg shadow-ocean-700/30",
    outline: "border-2 border-ocean-500 text-ocean-600 hover:bg-ocean-50"
  };

  const Component = href ? motion.a : motion.button;
  
  return (
    // @ts-ignore - motion component types are tricky with dynamic "as" behavior equivalent
    <Component
      href={href}
      className={`${baseStyles} ${variants[variant]} ${className}`}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      {...(props as any)}
    >
      {children}
    </Component>
  );
};
