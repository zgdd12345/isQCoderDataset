#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent提示词模板
"""


def get_instruction_prompt(language_mode: str) -> str:
    """
    根据语言模式返回指令生成提示词（只生成instruction，不生成input）

    Args:
        language_mode: "en" (英文), "zh" (纯中文)
    """
    if language_mode == "en":
        return """You are a professional quantum computing education expert. Please generate a valuable learning instruction based on the following paper segment.

Paper Title: {paper_title}
Paper Abstract:
{paper_abstract}
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
9. Do NOT include citation tags or source labels such as `` or 【来源: x】
10. The instruction must be entirely in English with no mixed Chinese
11. SELF-CONTAINMENT: The instruction must be understandable WITHOUT reading the original paper
   - Avoid vague references like "this algorithm", "the above method", "the system"
   - Use explicit names: "For the Quantum Phase Estimation algorithm..." instead of "For this algorithm..."
   - If referencing a mathematical object, include its definition: "Given the Hamiltonian H = ΣᵢZᵢZᵢ₊₁..."
12. The instruction should be answerable by an expert who only sees (instruction + input), not the segment

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
    # language_mode == "zh"
    return """你是一个专业的量子计算教育专家。请基于以下论文段落，生成一个有价值的学习指令。

论文标题：《{paper_title}》
论文摘要：
{paper_abstract}
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
9. 禁止出现任何引用标签或来源标记，如 ``、【来源: x】 等
10. 语言必须为纯中文，严禁中英混杂
11. 自包含性要求：指令必须在不阅读原始论文的情况下可理解
   - 避免模糊指代词，如"该算法"、"上述方法"、"这个系统"
   - 使用明确名称："对于量子相位估计算法..."而非"对于该算法..."
   - 引用数学对象时包含定义："给定哈密顿量H = ΣᵢZᵢZᵢ₊₁..."
12. 指令应该可由只看到(instruction + input)的专家回答，而无需段落内容

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
        language_mode: "en" (英文), "zh" (纯中文)
    """
    if language_mode == "en":
        return """You are a professional quantum computing expert. Please generate an accurate, professional, and detailed answer based on the following instruction and original paper segment.

Original Paper Title: {paper_title}
Original Paper Abstract:
{paper_abstract}
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
6. **IMPORTANT: The entire answer must be in English and match the instruction language; do not mix Chinese**
7. Do NOT include citation tags or source labels such as `` or 【来源: x】
8. Do NOT reference figures or tables that are not fully described in the input (avoid "see Fig. 1", "as shown in Table 2")
9. Use standard LaTeX delimiters $...$ or $$...$$ for all formulas (do not use \\(\\) or \\[\\])
10. SELF-CONTAINMENT: The answer must be comprehensible using ONLY (instruction + input)
   - Do NOT reference the paper structure: avoid "as mentioned above", "from section 3", "equation (5)"
   - If you need a formula, restate it fully in the answer (it may already be in input, but verify)
   - Define all notation used in your answer, even if defined in the segment
11. CONTEXT INDEPENDENCE: Assume the reader has NOT seen the original paper
   - Don't use phrases like "as we saw", "the previously discussed", "this system"
   - Be explicit: "In the quantum phase estimation algorithm discussed above..." → "The quantum phase estimation algorithm..."
12. **CRITICAL: NEVER mention the source paper or segment in your answer**
   - FORBIDDEN phrases: "Based on the provided segment", "from the paper", "according to the segment", "as stated in the original paper"
   - Your answer should present knowledge directly, as if you are teaching the concept yourself
   - Start with the topic directly, not with meta-references to where the information came from

Please return the answer content directly, no JSON format needed."""
    # language_mode == "zh"
    return """你是一个专业的量子计算专家。请根据以下指令和原始论文段落，生成一个准确、专业、详尽的回答。

原始论文标题：《{paper_title}》
原始论文摘要：
{paper_abstract}
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
6. 必须为纯中文回答，严禁中英混杂
7. 禁止出现任何引用标签或来源标记，如 ``、【来源: x】 等
8. 禁止引用未转化为文本的图/表（例如“见图1”“如表2所示”），除非表格内容已完整写入input
9. 所有数学公式统一使用标准LaTeX格式$...$或$$...$$，不要使用\\(\\)或\\[\\]
10. 自包含性：回答必须仅使用(instruction + input)即可理解
   - 不要引用论文结构：避免"如上所述"、"第3节中"、"公式(5)"
   - 需要公式时完整重述（可能已在input中，但需验证）
   - 定义回答中使用的所有符号，即使段落中已定义
11. 上下文独立性：假设读者未看过原始论文
   - 避免"如我们所见"、"前面讨论的"、"该系统"等短语
   - 明确表达："上文讨论的量子相位估计算法..." → "量子相位估计算法..."
12. **关键要求：绝对不要在回答中提及来源论文或段落**
   - 禁用短语："基于提供的片段"、"根据原始论文段落"、"从论文中"、"如原文所述"
   - 你的回答应该直接呈现知识，就像你在亲自教授这个概念
   - 直接从主题开始，不要引用信息来源

请直接返回回答内容，不需要JSON格式。"""


def get_input_prompt(language_mode: str) -> str:
    """
    根据语言模式返回input生成提示词

    Args:
        language_mode: "en" (英文), "zh" (纯中文)
    """
    if language_mode == "en":
        return """You are a professional instruction data quality expert. Please analyze the following instruction and determine whether it needs an input field, and if so, generate appropriate input content.

Original Paper Title: {paper_title}
Original Paper Abstract:
{paper_abstract}
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
5. CONTEXT EXTRACTION: When generating input, extract ALL necessary background from the segment:
   - Key term definitions that instruction references
   - Complete formula/equation definitions with variable explanations
   - Problem setup details (system size, boundary conditions, parameter ranges)
   - Physical interpretation of mathematical objects
6. SELF-CONTAINMENT TEST: Can an expert answer the instruction using ONLY (instruction + input) without seeing the segment?
   - If NO: add missing context to input
   - If YES: input is sufficient
7. **CRITICAL: If input is empty, the answer CANNOT reference "the segment" or "the paper"**
   - Empty input means ALL necessary context must already be in the instruction
   - If key definitions/formulas are needed but missing from instruction, you MUST add them to input
   - Test: Would an expert say "I need more context" when seeing only (instruction + empty_input)?
8. PURE INPUT: The input must be pure knowledge or data, with no meta prefixes like "Key context from paper:", "Based on segment:", or "Context:"
9. Avoid redundancy: Don't repeat what's already clear in the instruction
10. Keep input in English to match the instruction language; do not mix Chinese
11. Use standard LaTeX delimiters $...$ or $$...$$ for any formulas

Please return in JSON format:
{{
    "needs_input": true or false,
    "reasoning": "Brief explanation of why input is or isn't needed",
    "input": "The generated input content, or empty string if not needed"
}}"""
    # language_mode == "zh"
    return """你是一个专业的指令数据质量专家。请分析以下指令，判断是否需要input字段，如果需要则生成合适的input内容。

原始论文标题：《{paper_title}》
原始论文摘要：
{paper_abstract}
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
5. 上下文提取：生成input时，从段落中提取所有必要背景：
   - 指令中引用的关键术语定义
   - 完整的公式/方程定义及变量解释
   - 问题设置细节（系统规模、边界条件、参数范围）
   - 数学对象的物理意义
6. 自包含性测试：专家能否仅用(instruction + input)回答指令，而无需看段落？
   - 如果不能：将缺失上下文添加到input
   - 如果可以：input已足够
7. **关键要求：如果input为空，回答就不能引用"段落"或"论文"**
   - 空input意味着所有必要上下文必须已经在指令中
   - 如果关键定义/公式缺失但指令中未包含，你必须将它们添加到input
   - 测试：专家看到(指令+空input)时会不会说"我需要更多上下文"？
8. 纯净输入：input必须是纯粹的知识片段或数据，禁止包含如 "Key context from paper:"、"Based on segment:"、"Context:" 等元数据前缀
9. 避免冗余：不要重复指令中已经明确的内容
10. input必须使用中文，严禁中英混杂
11. 所有数学公式统一使用标准LaTeX格式$...$或$$...$$

请以JSON格式返回：
{{
    "needs_input": true或false,
    "reasoning": "简要说明为什么需要或不需要input",
    "input": "生成的input内容，如果不需要则为空字符串"
}}"""


VERIFICATION_PROMPT = """你是一个严格的学术审核专家。请验证以下生成的指令-回答对是否准确反映了原始论文的内容和本意。

原始论文标题：《{paper_title}》
原始论文摘要：
{paper_abstract}
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
6. 指令和输入是否一致，是否应该生成输入而未生成
7. 上下文泄漏检查（宽松模式）：
   - 检查是否存在未定义的指代词："该/这个/上述+算法/方法/系统"（需要有明确的先行词）
   - 检查是否引用了论文结构："如第X节所述"、"公式(编号)"、"如图/表X"
   - 检查是否使用了instruction/input中未定义的论文特定符号或术语
   - 注意：允许量子计算领域的标准术语（如qubit、Hamiltonian、entanglement、CNOT门等）
8. 自包含性评估：
   - 假设学习者只能看到(instruction, input, output)三个字段
   - 判断学习者是否能完全理解内容，无需查阅原始论文
   - 如果需要论文上下文才能理解，标记为未通过

请以严格JSON格式返回验证结果（仅输出JSON，不要Markdown代码块或额外解释；使用双引号；不要出现未转义的反斜杠）：
{{
    "passed": true或false,
    "confidence_score": 0.0到1.0的数值,
    "issues": ["发现的问题1", "发现的问题2"],
    "context_leakage": ["上下文泄漏实例1", "实例2"],
    "self_contained": true或false,
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
