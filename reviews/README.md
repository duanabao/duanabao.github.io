# Review Service Records

Private page for tracking academic review service.

**Access URL:** https://duanabao.github.io/reviews/

## Design Rules

1. **Directory Structure:**
   ```
   reviews/
   ├── index.html          # Main review page (by venue)
   ├── README.md           # This file
   └── {YEAR}/
       └── {VENUE}/
           ├── index.html  # Venue detail page
           └── {files}     # PDF files
   ```

2. **File Naming Convention:**
   - `{VENUE}{YEAR}-Paper{ID}.pdf` - Review confirmation
   - `{VENUE}{YEAR}-invitations.pdf` - Invitation email
   - `{ID}_invitation.pdf` - Journal invitation
   - `{ID}_review.pdf` - Review report
   - `{ID}_ack.pdf` - Acknowledgment

3. **Privacy:**
   - This page is NOT linked from the main website
   - Add `<meta name="robots" content="noindex, nofollow">` to all pages

## Review Statistics

### Conferences

| Year | Venue | Papers | Rank |
|------|-------|--------|------|
| 2026 | ICASSP | 12 | CCF-B |
| 2025 | ICASSP | 5 | CCF-B |

### Journals

| Year | Venue | Papers | IF/Rank |
|------|-------|--------|---------|
| 2026 | Scientific Data | 2 | IF 5.8, Q1 |

**Total Reviews: 19**
