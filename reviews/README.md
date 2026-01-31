# Review Service Records

Private page for tracking academic review service.

**Access URL:** https://duanabao.github.io/reviews/

## Design Rules

1. **Directory Structure:**
   ```
   reviews/
   ├── index.html          # Main review page
   ├── README.md           # This file
   └── {YEAR}/
       └── {VENUE}/
           ├── index.html  # Venue detail page
           └── {PAPER-ID}/ # Paper files (invitation, review, thanks, certificate)
   ```

2. **File Naming Convention:**
   - `{PAPER-ID}_invitation.pdf` - Review invitation from editor
   - `{PAPER-ID}_review.pdf` - Submitted review report
   - `{PAPER-ID}_thanks.pdf` - Acknowledgment email
   - `{PAPER-ID}_certificate.pdf` - Reviewer certificate

3. **Privacy:**
   - This page is NOT linked from the main website
   - Add `<meta name="robots" content="noindex, nofollow">` to all pages

## Review Statistics

| Year | Venue | Type | Papers | IF/Rank |
|------|-------|------|--------|---------|
| 2026 | ICASSP | Conference | 12 | CCF-B |
| 2026 | Scientific Data | Journal | 2 | IF 5.8, Q1 |

**Total Reviews: 14**
