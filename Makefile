all: quicktest

test: quicktest

checkforcyclicdependencies:
	pylint --disable=all --enable=cyclic-import src

removeunusedimports:
	pycln src

typecheck:
	# was: mypy --strict .
	mypy --explicit-package-bases --strict .

quicktest:
	export PYTHONPATH=. && pytest tests
	# Fail if radon scores any module (other than a few tolerated exceptions) "C" or worse
	radon cc -s --min C src | grep '^src' | grep -v factory | grep -v access_controller | grep -v to_dict > /tmp/radon.out || true
	@if [ -s /tmp/radon.out ]; then \
		cat /tmp/radon.out; \
		rm /tmp/radon.out; \
		exit 1; \
	fi
	@rm -f /tmp/radon.out

licenses:
	pip-licenses

licensereport:
	echo "# License list" > resources/reports/license-report-backend.txt
	./resources/scripts/generatelicenselist.sh >> resources/reports/license-report-backend.txt
	echo "" >> resources/reports/license-report-backend.txt
	echo "# Full report" >> resources/reports/license-report-backend.txt
	pip-licenses >> resources/reports/license-report-backend.txt
	echo "" >> resources/reports/license-report-backend.txt
	echo "# Dependency tree" >> resources/reports/license-report-backend.txt
	pipdeptree >> resources/reports/license-report-backend.txt

dependencytree:
	pipdeptree

