from templatey.interpolators import transform_unicode_control
from templatey.interpolators import untransform_unicode_control


class TestUnicodeControlTransforming:

    def test_round_trip_braces(self):
        text = 'something {with} brackets'
        assert '{' in text
        assert '}' in text

        transformed = transform_unicode_control(text)
        assert '{' not in transformed
        assert '}' not in transformed

        untransformed = untransform_unicode_control(transformed)
        assert untransformed == text

    def test_transform_and_format(self):
        text = 'something {with} brackets and ␎variable␏'

        transformed = transform_unicode_control(text)
        interpolated = transformed.format(variable='foo')

        untransformed = untransform_unicode_control(interpolated)
        assert untransformed == 'something {with} brackets and foo'
