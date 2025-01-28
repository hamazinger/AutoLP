from typing import List
import pandas as pd

from ..models import TitleAnalysis

class SeminarTitleEvaluator:
    def __init__(self, seminar_data: pd.DataFrame):
        self.df = seminar_data
        self._initialize_analytics()

    def _initialize_analytics(self):
        high_performing = self.df[self.df['Acquisition_Speed'] >= 2.5]
        self.attractive_words = self._extract_effective_keywords(high_performing)
        self.category_speeds = self.df.groupby('Major_Category')['Acquisition_Speed'].mean()
        self.problem_indicators = [
            '課題', '問題', 'による', 'ための', '向上', '改善', '解決', '対策',
            'どうする', 'なぜ', 'どう', '方法', '実現', 'ポイント', '実践',
            'ベストプラクティス', 'ノウハウ', '事例', '成功'
        ]

    def _extract_effective_keywords(self, high_performing_df) -> List[str]:
        words = []
        for title in high_performing_df['Seminar_Title']:
            if isinstance(title, str):
                clean_title = (title.replace('～', ' ')
                                 .replace('、', ' ')
                                 .replace('【', ' ')
                                 .replace('】', ' ')
                                 .replace('「', ' ')
                                 .replace('」', ' '))
                title_words = [w for w in clean_title.split()
                              if len(w) > 1 and not w.isdigit()]
                words.extend(title_words)

        word_counts = pd.Series(words).value_counts()
        return list(word_counts[word_counts >= 2].index)

    def _generate_evaluation_comment(self, analysis_data: dict) -> str:
        comments = []

        if analysis_data["predicted_speed"] >= 2.5:
            comments.append("高い集客が期待できます")
        elif analysis_data["predicted_speed"] >= 1.8:
            comments.append("一定の集客が見込めます")
        else:
            comments.append("改善の余地があります")

        if analysis_data["attractive_words"]:
            comments.append("効果的なキーワードが含まれています")
        else:
            comments.append("効果的なキーワードの追加を検討してください")

        if analysis_data["title_length"] > 40:
            comments.append("タイトルを短くすることを推奨します")

        if not analysis_data["has_specific_problem"]:
            comments.append("具体的な課題や問題提起の追加を検討してください")

        return "。".join(comments)

    def evaluate_title(self, title: str, category: str = None) -> TitleAnalysis:
        base_score = self._calculate_base_score(title)

        category_score = 0.0
        if category and category in self.category_speeds:
            category_avg = self.category_speeds[category]
            category_score = 0.3 if category_avg > 2.5 else (
                0.2 if category_avg > 2.0 else 0.1
            )

        final_score = min(max(base_score + category_score, 1.0), 3.0)

        matching_words = [word for word in self.attractive_words if word in title]
        has_problem = any(indicator in title for indicator in self.problem_indicators)

        reasoning = {
            "keywords": f"効果的なキーワード: {', '.join(matching_words) if matching_words else '該当なし'}",
            "title_length": f"タイトルの長さ: {len(title)}文字 （{'適切' if len(title) <= 40 else '長い'}）",
            "problem_indication": f"問題提起: {'あり' if has_problem else 'なし'}",
            "exclamation": f"感嘆符: {'あり（減点）' if '!' in title or '！' in title else 'なし'}",
            "category": f"カテゴリ評価: {category if category else '未指定'} (スコア: {category_score:.1f})",
            "predicted_speed": f"予測される集客速度: {final_score:.1f}"
        }

        grade = 'A' if final_score >= 2.5 else 'B' if final_score >= 1.8 else 'C'

        analysis_data = {
            "predicted_speed": final_score,
            "attractive_words": matching_words,
            "has_specific_problem": has_problem,
            "title_length": len(title)
        }

        evaluation_comment = self._generate_evaluation_comment(analysis_data)

        return TitleAnalysis(
            predicted_speed=final_score,
            grade=grade,
            attractive_words=matching_words,
            has_specific_problem=has_problem,
            has_exclamation='!' in title or '！' in title,
            title_length=len(title),
            category_score=category_score,
            reasoning=reasoning,
            evaluation_comment=evaluation_comment
        )

    def _calculate_base_score(self, title: str) -> float:
        base_score = 1.0

        matching_words = [word for word in self.attractive_words if word in title]
        keyword_score = len(matching_words) * 0.4
        base_score += min(keyword_score, 1.2)

        title_length = len(title)
        if title_length <= 20:
            base_score += 0.3
        elif title_length <= 40:
            base_score += 0.1
        elif title_length > 60:
            base_score -= 0.2

        if any(indicator in title for indicator in self.problem_indicators):
            base_score += 0.4

        if '!' in title or '！' in title:
            base_score -= 0.3

        return base_score