import re
from typing import Dict, List

class TextProcessor:
    def __init__(self):
        self.stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'can', 'a', 'an', 'this', 'that', 'these', 'those'
        }
        
        self.section_keywords = {
            'skills': ['skills', 'technical skills', 'technologies', 'tools', 'expertise'],
            'experience': ['experience', 'work experience', 'employment', 'career'],
            'education': ['education', 'academic', 'degree', 'university', 'college'],
            'projects': ['projects', 'portfolio', 'work samples']
        }
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract different sections from resume text"""
        sections = {}
        text_lower = text.lower()
        
        for section_name, keywords in self.section_keywords.items():
            section_text = ""
            for keyword in keywords:
                pattern = rf'{keyword}[:\-\s]*(.*?)(?=\n[A-Z]|\n\n|\Z)'
                matches = re.findall(pattern, text_lower, re.DOTALL | re.IGNORECASE)
                if matches:
                    section_text += " ".join(matches)
            
            sections[section_name] = section_text.strip()
        
        return sections
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text"""
        common_skills = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node.js',
            'django', 'flask', 'spring', 'hibernate', 'sql', 'mysql', 'postgresql',
            'mongodb', 'redis', 'elasticsearch', 'aws', 'azure', 'gcp', 'docker',
            'kubernetes', 'jenkins', 'git', 'linux', 'windows', 'agile', 'scrum',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
            'data analysis', 'pandas', 'numpy', 'matplotlib', 'tableau', 'power bi'
        ]
        
        found_skills = []
        text_lower = text.lower()
        
        for skill in common_skills:
            if skill in text_lower:
                found_skills.append(skill)
        
        return found_skills