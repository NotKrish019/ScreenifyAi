from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class SimpleMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    def rank_resumes(self, jd_text, resumes):
        # resumes is a list of dicts: {'id': ..., 'name': ..., 'original_text': ...}
        
        # 1. Prepare corpus
        clean_jd = self.clean_text(jd_text)
        clean_resumes = [self.clean_text(r['original_text']) for r in resumes]
        
        corpus = [clean_jd] + clean_resumes
        
        # 2. Vectorize
        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
        except ValueError:
            # Handle empty vocabulary case
            return []

        # 3. Calculate similarity (first vector is JD)
        jd_vector = tfidf_matrix[0]
        resume_vectors = tfidf_matrix[1:]
        
        # cosine_similarity with JD vector
        similarities = cosine_similarity(jd_vector, resume_vectors).flatten()
        
        # 4. Format results
        results = []
        for i, score in enumerate(similarities):
            # Scale to 0-100
            match_score = round(score * 100, 1)
            
            # Simple fit categorization
            if match_score >= 70: fit = "High"
            elif match_score >= 40: fit = "Medium"
            else: fit = "Low"
            
            res_data = resumes[i]
            
            results.append({
                "id": res_data['id'],
                "resume_name": res_data['name'],
                "match_score": match_score,
                "fit": fit,
                "summary": f"Similarity score: {match_score}% based on keyword matching.",
                "matched_skills": [], # We can implement simple extraction if needed, but keeping it simple for now
                "missing_skills": [],
                "explanation": {}
            })
            
        # Sort by score descending
        results.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Deduplicate based on base filename (e.g., "resume.pdf" and "resume.txt" -> "resume")
        unique_results = []
        seen_names = set()

        for r in results:
            # Simple way to get filename without extension
            base_name = r['resume_name'].rsplit('.', 1)[0]
            
            if base_name not in seen_names:
                seen_names.add(base_name)
                unique_results.append(r)
        
        # Add rank
        for i, r in enumerate(unique_results):
            r['rank'] = i + 1
            
        return unique_results

simple_matcher = SimpleMatcher()
