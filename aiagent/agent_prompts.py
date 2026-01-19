#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent提示词模板
"""


def get_instruction_prompt(language_mode: str) -> str:
    """
    根据语言模式返回指令生成提示词（只生成instruction，不生成input）

    Args:
        language_mode: "en" (英文), "mixed" (中英混合), "zh" (纯中文)
    """
    if language_mode == "en":
        return """You are a professional quantum computing education expert. Please generate a valuable learning instruction based on the following paper segment.

Paper Title: {paper_title}
Segment Title: {segment_title}
Segment Content:
{segment_content}

Requirements:
1. Carefully analyze the core knowledge points and key concepts in the segment
2. Generate a clear and specific instruction that can guide learners to deeply understand this content
3. The instruction should be a question or task, not a simple paraphrase of the content
4. Keep instruction in English; do not mix languages
5. If the segment is not suitable for generating an instruction, return an empty string for instruction
6. Use standard LaTeX delimiters $...$ or $$...$$ for any formulas (do not use \\(\\) or \\[\\])
7. Do NOT generate questions that analyze graphs or tables
8. Focus ONLY on generating the instruction - do NOT generate input field (it will be handled separately)

Instruction Type Options:
- concept: Concept explanation and definition
- implementation: Method steps or algorithm implementation
- analysis: Problem analysis and discussion
- comparison: Comparative analysis
- application: Application scenario introduction
- others: Other related types

Please return in JSON format:
{{
    "instruction": "The generated instruction content IN ENGLISH",
    "instruction_type": "instruction type",
    "key_concepts": ["key concept 1", "key concept 2"]
}}"""
    if language_mode == "mixed":
        return """你是一个专业的量子计算教育专家。请基于以下论文段落，生成一个有价值的学习指令。

论文标题：《{paper_title}》
段落标题：{segment_title}
段落内容：
{segment_content}

要求：
1. 仔细分析段落中的核心知识点和关键概念
2. 生成一个清晰、具体的指令，这个指令应该能够引导学习者深入理解这段内容
3. 指令应该是一个问题或任务，而不是对内容的简单复述
4. **IMPORTANT: 指令必须使用中文**
5. 选择一个合适的指令类型
6. 如果该段落不适合生成指令，请返回空字符串
7. 所有数学公式统一使用标准LaTeX格式$...$或$$...$$，不要使用\\(\\)或\\[\\]
8. 禁止生成分析图或者表格的问题
9. 只生成instruction，不要生成input字段（input将由单独步骤处理）

指令类型选项：
- concept: 概念解释和定义
- implementation: 方法步骤或算法实现
- analysis: 问题分析和讨论
- comparison: 比较分析
- application: 应用场景介绍
- others: 其他相关类型

请以JSON格式返回：
{{
    "instruction": "生成的指令内容（中文）",
    "instruction_type": "指令类型",
    "key_concepts": ["关键概念1", "关键概念2"]
}}"""
    return """你是一个专业的量子计算教育专家。请基于以下论文段落，生成一个有价值的学习指令。

论文标题：《{paper_title}》
段落标题：{segment_title}
段落内容：
{segment_content}

要求：
1. 仔细分析段落中的核心知识点和关键概念
2. 生成一个清晰、具体的指令，这个指令应该能够引导学习者深入理解这段内容
3. 指令应该是一个问题或任务，而不是对内容的简单复述
4. 指令必须使用中文
5. 选择一个合适的指令类型
6. 如果该段落不适合生成指令，请返回空字符串
7. 所有数学公式统一使用标准LaTeX格式$...$或$$...$$，不要使用\\(\\)或\\[\\]
8. 只生成instruction，不要生成input字段（input将由单独步骤处理）

指令类型选项：
- concept: 概念解释和定义
- implementation: 方法步骤或算法实现
- analysis: 问题分析和讨论
- comparison: 比较分析
- application: 应用场景介绍
- others: 其他相关类型

请以JSON格式返回：
{{
    "instruction": "生成的指令内容",
    "instruction_type": "指令类型",
    "key_concepts": ["关键概念1", "关键概念2"]
}}"""


def get_answer_prompt(language_mode: str) -> str:
    """
    根据语言模式返回回答生成提示词

    Args:
        language_mode: "en" (英文), "mixed" (中英混合), "zh" (纯中文)
    """
    if language_mode == "en":
        return """You are a professional quantum computing expert. Please generate an accurate, professional, and detailed answer based on the following instruction and original paper segment.

Original Paper Title: {paper_title}
Original Segment Title: {segment_title}
Original Segment Content:
{segment_content}

Instruction: {instruction}
Input: {input}
Instruction Type: {instruction_type}

Requirements:
1. The answer must be based on the original segment content to ensure accuracy
2. The answer should be comprehensive and professional, helping learners truly understand the relevant knowledge
3. You can expand and explain appropriately, but the core content must come from the original text
4. If it involves formulas or algorithms, please explain their meanings and derivation processes in detail
5. Use a clear structure to organize the answer (you can use numbering, bullet points, etc.)
6. **IMPORTANT: The entire answer must be in English and match the instruction language**
7. Use standard LaTeX delimiters $...$ or $$...$$ for all formulas (do not use \\(\\) or \\[\\])

Please return the answer content directly, no JSON format needed."""
    if language_mode == "mixed":
        return """你是一个专业的量子计算专家。请根据以下指令和原始论文段落，生成一个准确、专业、详尽的回答。

原始论文标题：《{paper_title}》
原始段落标题：{segment_title}
原始段落内容：
{segment_content}

指令：{instruction}
输入：{input}
指令类型：{instruction_type}

要求：
1. 回答必须基于原始段落的内容，确保准确性
2. 回答应该全面、专业，帮助学习者真正理解相关知识
3. 可以适当扩展和解释，但核心内容必须来自原文
4. 如果涉及公式或算法，请详细解释其含义和推导过程
5. 使用清晰的结构组织回答（可使用编号、分点等）
6. **IMPORTANT: 回答语言必须与指令一致（中文）**
7. 所有数学公式统一使用标准LaTeX格式$...$或$$...$$，不要使用\\(\\)或\\[\\]

请直接返回回答内容，不需要JSON格式。"""
    return """你是一个专业的量子计算专家。请根据以下指令和原始论文段落，生成一个准确、专业、详尽的回答。

原始论文标题：《{paper_title}》
原始段落标题：{segment_title}
原始段落内容：
{segment_content}

指令：{instruction}
输入：{input}
指令类型：{instruction_type}

要求：
1. 回答必须基于原始段落的内容，确保准确性
2. 回答应该全面、专业，帮助学习者真正理解相关知识
3. 可以适当扩展和解释，但核心内容必须来自原文
4. 如果涉及公式或算法，请详细解释其含义和推导过程
5. 使用清晰的结构组织回答（可使用编号、分点等）
6. 所有数学公式统一使用标准LaTeX格式$...$或$$...$$，不要使用\\(\\)或\\[\\]

请直接返回回答内容，不需要JSON格式。"""


def get_input_prompt(language_mode: str) -> str:
    """
    根据语言模式返回input生成提示词

    Args:
        language_mode: "en" (英文), "mixed" (中英混合), "zh" (纯中文)
    """
    if language_mode == "en":
        return """You are a professional instruction data quality expert. Please analyze the following instruction and determine whether it needs an input field, and if so, generate appropriate input content.

Original Paper Title: {paper_title}
Original Segment Title: {segment_title}
Original Segment Content:
{segment_content}

Generated Instruction: {instruction}
Instruction Type: {instruction_type}

Analysis Guidelines:
1. Determine if this instruction requires additional context, constraints, parameters, or data to be answered properly
2. An input field is NEEDED when:
   - The instruction references specific parameters, thresholds, or configurations that should be specified
   - The instruction asks about a specific scenario that needs context
   - The instruction requires specific data, code snippets, or examples to work with
   - The instruction mentions "given", "for", "with" implying external input
3. An input field is NOT NEEDED when:
   - The instruction is self-contained and can be answered directly
   - The instruction asks for general explanations or definitions
   - All necessary context is already in the instruction itself
4. If input is needed, generate concise but complete input content that:
   - Provides necessary context without being redundant with the instruction
   - Uses concrete examples or values when appropriate
   - Is relevant to the paper segment content
5. Keep input in English to match the instruction language
6. Use standard LaTeX delimiters $...$ or $$...$$ for any formulas

Please return in JSON format:
{{
    "needs_input": true or false,
    "reasoning": "Brief explanation of why input is or isn't needed",
    "input": "The generated input content, or empty string if not needed"
}}"""
    if language_mode == "mixed":
        return """你是一个专业的指令数据质量专家。请分析以下指令，判断是否需要input字段，如果需要则生成合适的input内容。

原始论文标题：《{paper_title}》
原始段落标题：{segment_title}
原始段落内容：
{segment_content}

生成的指令：{instruction}
指令类型：{instruction_type}

分析指南：
1. 判断这个指令是否需要额外的上下文、约束、参数或数据才能被正确回答
2. 以下情况需要input字段：
   - 指令涉及特定的参数、阈值或配置需要指定
   - 指令询问需要上下文的特定场景
   - 指令需要特定的数据、代码片段或示例
   - 指令中提到"给定"、"对于"、"在...条件下"等暗示外部输入
   - 指令中提到"根据图"或"根据表"等类似内容，需要补充图表的数据或说明
3. 以下情况不需要input字段：
   - 指令是自包含的，可以直接回答
   - 指令询问一般性的解释或定义
   - 所有必要的上下文已经在指令本身中
4. 如果需要input，生成简洁但完整的input内容：
   - 提供必要的上下文，但不与指令重复
   - 在适当时使用具体的示例或数值
   - 与论文段落内容相关
5. **input必须使用中文**（与指令语言一致）
6. 所有数学公式统一使用标准LaTeX格式$...$或$$...$$

请以JSON格式返回：
{{
    "needs_input": true或false,
    "reasoning": "简要说明为什么需要或不需要input",
    "input": "生成的input内容（中文），如果不需要则为空字符串"
}}"""
    return """你是一个专业的指令数据质量专家。请分析以下指令，判断是否需要input字段，如果需要则生成合适的input内容。

原始论文标题：《{paper_title}》
原始段落标题：{segment_title}
原始段落内容：
{segment_content}

生成的指令：{instruction}
指令类型：{instruction_type}

分析指南：
1. 判断这个指令是否需要额外的上下文、约束、参数或数据才能被正确回答
2. 以下情况需要input字段：
   - 指令涉及特定的参数、阈值或配置需要指定
   - 指令询问需要上下文的特定场景
   - 指令需要特定的数据、代码片段或示例
   - 指令中提到"给定"、"对于"、"在...条件下"等暗示外部输入
3. 以下情况不需要input字段：
   - 指令是自包含的，可以直接回答
   - 指令询问一般性的解释或定义
   - 所有必要的上下文已经在指令本身中
4. 如果需要input，生成简洁但完整的input内容：
   - 提供必要的上下文，但不与指令重复
   - 在适当时使用具体的示例或数值
   - 与论文段落内容相关
5. input必须使用中文
6. 所有数学公式统一使用标准LaTeX格式$...$或$$...$$

请以JSON格式返回：
{{
    "needs_input": true或false,
    "reasoning": "简要说明为什么需要或不需要input",
    "input": "生成的input内容，如果不需要则为空字符串"
}}"""


VERIFICATION_PROMPT = """你是一个严格的学术审核专家。请验证以下生成的指令-回答对是否准确反映了原始论文的内容和本意。

原始论文标题：《{paper_title}》
原始论文相关段落：
{segment_content}

生成的指令：{instruction}
生成的输入：{input}
生成的回答：{output}

请严格检查以下几点：
1. 准确性：回答中的事实、公式、概念是否与原文一致
2. 完整性：回答是否涵盖了指令所问的关键点
3. 一致性：回答的观点和论述是否与原文的立场一致
4. 无幻觉：是否存在原文未提及的信息被当作事实陈述
5. 语言表达：回答是否清晰、专业，符合学术规范
6. 指令和输入是否一致，是否应该生成输入而未生成。

请以JSON格式返回验证结果：
{{
    "passed": true或false,
    "confidence_score": 0.0到1.0的数值,
    "issues": ["发现的问题1", "发现的问题2"],
    "suggestion": "如果未通过，建议如何修改"
}}"""


SEGMENT_RELEVANCE_PROMPT = """你是一个严格的学术内容筛选助手。请判断下面段落是否实质性讨论量子计算相关问题（如量子算法、量子电路、量子信息、量子硬件、误差校正、量子复杂度等）。
如果段落属于目录、致谢、参考文献、作者信息，或仅提到“量子”但没有实质讨论，或与量子计算无关，请返回false；若不确定，也返回false。

论文标题：{paper_title}
段落标题：{segment_title}
段落内容：
{segment_content}

请仅返回JSON：
{{
    "is_quantum": true或false,
    "reason": "简要原因"
}}"""
