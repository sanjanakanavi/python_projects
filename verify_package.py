import importlib.util

package_name = 'seaborn'  # Replace with the package you want to check

if importlib.util.find_spec(package_name) is not None:
    print(f"'{package_name}' is installed")
else:
    print(f"'{package_name}' is NOT installed")