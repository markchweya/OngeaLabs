self.__MIDDLEWARE_MATCHERS = [
  {
    "regexp": "^\\/OngeaLabs(?:\\/(_next\\/data\\/[^/]{1,}))?(?:\\/((?!_next|favicon.ico|assets|.*\\..*).*))(\\.json|\\.rsc|\\.segments\\/.+\\.segment\\.rsc)?[\\/#\\?]?$",
    "originalSource": "/((?!_next|favicon.ico|assets|.*\\..*).*)"
  },
  {
    "regexp": "^\\/OngeaLabs(?:\\/(_next\\/data\\/[^/]{1,}))?\\/robots\\.txt(\\.json|\\.rsc|\\.segments\\/.+\\.segment\\.rsc)?[\\/#\\?]?$",
    "originalSource": "/robots.txt"
  },
  {
    "regexp": "^\\/OngeaLabs(?:\\/(_next\\/data\\/[^/]{1,}))?\\/sitemap\\.xml(\\.json|\\.rsc|\\.segments\\/.+\\.segment\\.rsc)?[\\/#\\?]?$",
    "originalSource": "/sitemap.xml"
  },
  {
    "regexp": "^\\/OngeaLabs(?:\\/(_next\\/data\\/[^/]{1,}))?\\/favicon\\.ico(\\.json|\\.rsc|\\.segments\\/.+\\.segment\\.rsc)?[\\/#\\?]?$",
    "originalSource": "/favicon.ico"
  }
];self.__MIDDLEWARE_MATCHERS_CB && self.__MIDDLEWARE_MATCHERS_CB()