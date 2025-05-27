from templatey.prebaked.env_funcs import xml_attrify


class TestXmlAttrify:

    def test_noattr(self):
        """Passing no attrs must result in an empty result.
        """
        result = xml_attrify({})
        assert not result

        result = xml_attrify({}, trailing_space=True)
        assert not result

    def test_oneattr(self):
        """Passing a single attr must not add extra undesired spaces.
        """
        result = xml_attrify({'foo': 'oof'})
        assert ''.join(result) == 'foo="oof"'

        result = xml_attrify({'foo': 'oof'}, trailing_space=True)
        assert ''.join(result) == 'foo="oof" '

    def test_threeattr(self):
        """Passing three attrs must add correct spacing as desired.
        """
        result = xml_attrify({'foo': 'oof', 'bar': 'rab', 'baz': 'zab'})
        assert ''.join(result) == 'foo="oof" bar="rab" baz="zab"'

        result = xml_attrify(
            {'foo': 'oof', 'bar': 'rab', 'baz': 'zab'}, trailing_space=True)
        assert ''.join(result) == 'foo="oof" bar="rab" baz="zab" '

        result = xml_attrify(
            {'foo': 'oof', 'bar': 'rab', 'baz': 'zab'},
            interstitial_space=False)
        assert ''.join(result) == 'foo="oof"bar="rab"baz="zab"'

        result = xml_attrify(
            {'foo': 'oof', 'bar': 'rab', 'baz': 'zab'},
            trailing_space=True,
            interstitial_space=False)
        assert ''.join(result) == 'foo="oof"bar="rab"baz="zab" '
