#!/bin/sh
if command -v pylint > /dev/null; then
    pylint app/ app.py
    if [ $? -eq 0 ]; then
        echo "###################"
        echo "Quality test passed"
        echo "###################"
    else
        echo "###################"
        echo "Quality test fail"
        echo "###################"
        exit
    fi
else
    echo "###################"
    echo "pylint kurulu degil, atlandi"
    echo "###################"
fi

if [ -d tests ]; then
    pytest
    if [ $? -eq 0 ]; then
        echo "###################"
        echo "Unit test passed"
        echo "###################"
    else
        echo "###################"
        echo "Unit test fail"
        echo "###################"
        exit
    fi
else
    echo "###################"
    echo "tests/ yok, atlandi"
    echo "###################"
fi

python app.py --env local
if [ $? -eq 0 ]; then
    echo "###################"
    echo "end2end test passed."
    echo "###################"
else
    echo "###################"
    echo "end2end test fail."
    echo "###################"
    exit
fi

if [ -f Dockerfile ]; then
    docker build --tag project_name .
    if [ $? -eq 0 ]; then
        echo "###################"
        echo "Docker build test passed."
        echo "###################"
    else
        echo "###################"
        echo "Docker build test fail."
        echo "###################"
        exit
    fi
else
    echo "###################"
    echo "Dockerfile yok, atlandi"
    echo "###################"
fi
