import React from 'react';

interface SectionProps {
  id?: string;
  children: React.ReactNode;
  className?: string;
  background?: 'light' | 'dark' | 'gradient';
}

export const Section: React.FC<SectionProps> = ({
  id,
  children,
  className = '',
  background = 'light'
}) => {
  const backgrounds = {
    light: "bg-white",
    dark: "bg-black text-white",
    gradient: "bg-white"
  };

  return (
    <section id={id} className={`py-20 px-4 sm:px-6 lg:px-8 ${backgrounds[background]} ${className}`}>
      <div className="max-w-7xl mx-auto">
        {children}
      </div>
    </section>
  );
};
