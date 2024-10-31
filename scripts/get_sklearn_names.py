import importlib

if __name__ == '__main__':
    # Import the module
    module = importlib.import_module("pytabkit.models.sklearn.sklearn_interfaces")

    # Get all top-level attributes of the module (like classes, functions)
    attrs = [attr_name for attr_name in dir(module)
             if not attr_name.startswith('_') and not 'Mixin' in attr_name
             and hasattr(getattr(module, attr_name), '__module__')
             and getattr(module, attr_name).__module__ == module.__name__]
    print(f', '.join(attrs))
