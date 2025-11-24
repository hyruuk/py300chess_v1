"""
Test that all Python files have valid syntax.
This test doesn't import the modules, just checks syntax.
"""

import os
import py_compile
import sys

def test_syntax():
    """Test syntax of all Python files."""
    src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')

    print("Testing Python syntax...")
    errors = []
    success_count = 0

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, src_dir)

                try:
                    py_compile.compile(filepath, doraise=True)
                    print(f"✓ {rel_path}")
                    success_count += 1
                except py_compile.PyCompileError as e:
                    print(f"✗ {rel_path}")
                    errors.append((rel_path, str(e)))

    print(f"\n{success_count} files checked")

    if errors:
        print(f"\n❌ {len(errors)} files with syntax errors:")
        for path, error in errors:
            print(f"  {path}: {error}")
        return False
    else:
        print("\n✅ All files have valid syntax!")
        return True

if __name__ == '__main__':
    success = test_syntax()
    sys.exit(0 if success else 1)
