import py_compile, traceback
files = [
    r'D:\研究生\智能体\gopt_charsiu\src\prep_data\build_charsiu_seq_data.py',
    r'D:\研究生\智能体\gopt_charsiu\src\prep_data\build_whisper_prefix_data.py',
    r'D:\研究生\智能体\gopt_charsiu\src\prep_data\build_streaming_charsiu_data.py',
    r'D:\研究生\智能体\gopt_charsiu\src\prep_data\build_streaming_asr_gopt_data.py',
]
with open(r'D:\研究生\智能体\gopt_charsiu\_compile_check.out', 'w', encoding='utf-8') as out:
    for path in files:
        try:
            py_compile.compile(path, doraise=True)
            out.write('OK ' + path + '\n')
        except Exception:
            out.write('FAIL ' + path + '\n')
            out.write(traceback.format_exc() + '\n')
