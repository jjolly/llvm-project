from pygments.lexer import RegexLexer, bygroups
from pygments.token import *

class MLIRFuncArgs(RegexLexer):
    tokens = {
        'root': [
            (r'(\s*?)(%[a-zA-Z]\w*)(\s*?)(:)', bygroups(Whitespace, Name.Variable, Whitespace, Keyword), 'type')
        ]
        'type': [
            (r'\s*', Whitespace),
            (r'f32|f64|i8|i16|i32|i64', Keyword.Type),
            (r'(tensor)(\s*?)(<)', bygroups(Keyword.Type, Whitespace, ))
        ]
    }

class MLIRLexer(RegexLexer):
    name    = 'MLIR'
    aliases = ['mlir']
    filenames = ['*.mlir']
    
    tokens = {
        'root': [
            (r'//.*?$', Comment.Singleline),
            (r'#[a-zA-Z][.\w]*', Keyword),
            (r'=', Operator),
            (r'(func)(\s+)(@[a-zA-Z]\w*?)(\s*?)(\()([^\)]*?)(\))', bygroups(Keyword, Whitespace, Name.Function, Whitespace, Punctuation, Text, Punctuation)),
            (r'.', Text)
        ]
    }
