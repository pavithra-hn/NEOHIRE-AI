import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import pandas as pd

class ResumeMatcher:
    def __init__(self, embedding_generator):
        self.embedding_generator = embedding_generator
        self.similarity_threshold = 0.3
    
    def calculate_similarity(self, resume_embedding: np.ndarray, 
                           jd_embedding: np.ndarray) -> float:
        """Calculate cosine similarity between resume and job description"""
        if resume_embedding.ndim == 1:
            resume_embedding = resume_embedding.reshape(1, -1)
        if jd_embedding.ndim == 1:
            jd_embedding = jd_embedding.reshape(1, -1)
        
        similarity = cosine_similarity(resume_embedding, jd_embedding)[0][0]
        return float(similarity)
    
    def match_resume_to_jd(self, resume_text: str, jd_text: str) -> Dict:
        """Match a single resume to a job description"""
        # Generate embeddings
        resume_embedding = self.embedding_generator.generate_embeddings(resume_text)
        jd_embedding = self.embedding_generator.generate_embeddings(jd_text)
        
        # Calculate similarity
        similarity_score = self.calculate_similarity(resume_embedding, jd_embedding)
        
        # Generate match analysis
        match_analysis = self._analyze_match(resume_text, jd_text, similarity_score)
        
        return {
            'similarity_score': similarity_score,
            'match_percentage': similarity_score * 100,
            'match_level': self._get_match_level(similarity_score),
            'analysis': match_analysis
        }
    
    def batch_match_resumes(self, resumes: List[Dict], jd_text: str) -> List[Dict]:
        """Match multiple resumes to a job description"""
        results = []
        jd_embedding = self.embedding_generator.generate_embeddings(jd_text)
        
        for i, resume in enumerate(resumes):
            resume_embedding = self.embedding_generator.generate_embeddings(resume['text'])
            similarity_score = self.calculate_similarity(resume_embedding, jd_embedding)
            
            result = {
                'resume_name': resume.get('name', f'Resume_{i+1}'),
                'similarity_score': similarity_score,
                'match_percentage': similarity_score * 100,
                'match_level': self._get_match_level(similarity_score),
                'resume_text': resume['text'][:500] + '...' if len(resume['text']) > 500 else resume['text']
            }
            results.append(result)
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results
    
    def _get_match_level(self, similarity_score: float) -> str:
        """Categorize match level based on similarity score"""
        if similarity_score >= 0.7:
            return "Excellent Match"
        elif similarity_score >= 0.5:
            return "Good Match"
        elif similarity_score >= 0.3:
            return "Fair Match"
        else:
            return "Poor Match"
    
    def _analyze_match(self, resume_text: str, jd_text: str, 
                      similarity_score: float) -> Dict:
        """Provide detailed analysis of the match"""
        analysis = {
            'strengths': [],
            'gaps': [],
            'recommendations': []
        }
        
        resume_words = set(resume_text.lower().split())
        jd_words = set(jd_text.lower().split())
        
        # Find common keywords
        common_words = resume_words.intersection(jd_words)
        missing_words = jd_words - resume_words
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        common_words = common_words - stop_words
        missing_words = missing_words - stop_words
        
        # Generate analysis
        if similarity_score >= 0.5:
            analysis['strengths'] = [
                f"Strong keyword overlap: {', '.join(list(common_words)[:5])}",
                "Good semantic alignment with job requirements"
            ]
        
        if missing_words:
            analysis['gaps'] = [
                f"Missing key terms: {', '.join(list(missing_words)[:5])}"
            ]
        
        # Recommendations
        if similarity_score < 0.5:
            analysis['recommendations'] = [
                "Consider highlighting relevant experience more prominently",
                "Include more specific technical skills mentioned in the job description",
                "Adjust resume language to better match job requirements"
            ]
        
        return analysis
    
    def export_results_to_excel(self, results: List[Dict], filename: str = "resume_analysis_results.xlsx"):
        """Export results to Excel file"""
        df = pd.DataFrame(results)
        
        # Create Excel file with formatting
        with pd.ExcelWriter(f"outputs/results/{filename}", engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
            
            # Get workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Results']
            
            # Format headers
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)
                cell.fill = cell.fill.copy(fgColor="CCCCCC")
        
        return f"outputs/results/{filename}"