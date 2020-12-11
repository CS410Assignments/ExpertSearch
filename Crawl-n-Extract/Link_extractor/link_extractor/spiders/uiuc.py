# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractor import LinkExtractor
from scrapy.spiders import Rule, CrawlSpider
from link_extractor.items import LinkExtractorItem

class UiucSpider(CrawlSpider):
    # define name of the spider

    name = 'uiuc'

    # define allowed-domains and the web-url to crawl
    allowed_domains = ['utexas.edu']
    start_urls = ['https://www.cs.utexas.edu/']

    # set follow = False to limit the crawl for web links

    rules = [
            Rule(
                LinkExtractor(
                    canonicalize=True,
                    unique=True
                ),
                follow=False,
                callback="parse_items"
            )
        ]

    # Method which starts the requests by visiting all URLs specified in start_urls
    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse, dont_filter=False)

    # Method for parsing items
    def parse_items(self, response):
        # The list of items that are found on the particular page
        items = []
        # Only extract canonicalized and unique links (with respect to the current page)
        links = LinkExtractor(canonicalize=True,unique=True).extract_links(response)
        # Now go through all the found links
        for link in links:
            # Check whether the domain of the URL of the link is allowed; so whether it is in one of the allowed domains
            is_allowed = True
            for allowed_domain in self.allowed_domains:
                if allowed_domain in link.url:
                    is_allowed = True
            # If it is allowed, create a new item and add it to the list of found items
            if is_allowed:
                item = LinkExtractorItem()
                item['url_from'] = response.url
                item['url_to'] = link.url
                items.append(item)
    # Return all the found items
        return(items)